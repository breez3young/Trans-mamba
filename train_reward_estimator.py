import argparse
import torch
import pickle
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader

from networks.dreamer.reward_estimator import Reward_estimator
from agent.models.vq import SimpleVQAutoEncoder, SimpleFSQAutoEncoder

import ipdb

def symlog(x: np.ndarray):
    return np.sign(x) * np.log(np.abs(x) + 1)


#### Define reward dataset
class ObsRewardDataset(Dataset):
    def __init__(self, data):
        valid_indices = np.argwhere(data["fakes"].all(-2).squeeze() == False).squeeze().tolist()
        self.rewards = data['rewards'][valid_indices]
        self.observations = data['observations'][valid_indices]
        self.obs_dim = self.observations.shape[-1]
        self.n_agents = self.observations.shape[-2]
        
        assert self.rewards.shape[0] == self.observations.shape[0]
        
        print(f"Load {self.rewards.shape[0]} steps.")
        print(f"Before transformation, reward maximum: {self.rewards.max()}, minimum: {self.rewards.min()}")

        self.rewards = symlog(self.rewards)

        print(f"After transformation, reward maximum: {self.rewards.max()}, minimum: {self.rewards.min()}")
    
    def __len__(self):
        return self.rewards.shape[0]
    
    def __getitem__(self, index):
        return self.observations[index], self.rewards[index].mean(-2)
    
    
#### ---------------------



def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--mode', type=str, default='disabled')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    
    return parser

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Load Offline Dataset
    data_path = "/mnt/data/optimal/zhangyang/.offline_dt/mamba_50k.pkl"
    with open(data_path, 'rb+') as f:
        data = pickle.load(f)
    dataset = ObsRewardDataset(data)
    
    ## random split train and test dataset
    generator = torch.Generator().manual_seed(123)
    train_len = int(len(dataset) * 0.9)
    train_dataset, test_dataset = random_split(dataset, [train_len, len(dataset) - train_len], generator=generator)
    
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # 2. Initialize model and optimizer
    tokenizer = SimpleVQAutoEncoder(in_dim=dataset.obs_dim, embed_dim=32, num_tokens=16,
                                    codebook_size=512, learnable_codebook=False, ema_update=True, decay=0.8).to(device)
    
    model = Reward_estimator(in_dim=dataset.obs_dim,
                             hidden_size=256, n_agents=dataset.n_agents).to(device)
    initialize_weights(model, mode='xavier')

    lr = 3e-4
    params = list(tokenizer.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    
    # 3. Train reward model
    best_loss = float('inf')

    for epoch in range(args.epochs):
        
        ## Training
        model.train()
        tokenizer.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        loss_aver = 0.
        for it, (obs, rew) in pbar:
            obs = obs.to(device)
            rew = rew.to(device)
            
            out, indices, cmt_loss = tokenizer(obs, True, True)
            pred_rew = model(out)
            
            # loss = F.smooth_l1_loss(pred_rew, rew)
            rec_loss = (out - obs).abs().mean()
            rew_loss = F.smooth_l1_loss(pred_rew, rew)
            
            active_rate = indices.detach().unique().numel() / 512 * 100
            
            loss = rec_loss + rew_loss + 10. * cmt_loss

            ### Back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch + 1} it {it}: "
                                 + f"rec loss {rec_loss.item():.4f} | "
                                 + f"rew loss {rew_loss.item():.4f} | "
                                 + f"cmt loss {cmt_loss.item():.4f} | "
                                 + f"active %: {active_rate}")

            loss_aver += loss.item()
        
        print(f"Average loss: {(loss_aver / len(train_loader)):.4f}")

        ## Evaluation
        model.eval()
        tokenizer.eval()
        
        loss_eval_aver = 0.
        for it, (obs, rew) in enumerate(test_loader):
            with torch.no_grad():
                obs = obs.to(device)
                rew = rew.to(device)
                
                rec = tokenizer.encode_decode(obs, True, True)
                pred_rew = model(rec)
                
                eval_loss = F.smooth_l1_loss(pred_rew, rew) + (rec - obs).abs().mean()

                loss_eval_aver += eval_loss.item()

        print(f"Evaluation average loss: {(loss_eval_aver / len(test_loader)):.4f}")

        if loss_eval_aver < best_loss:
            best_loss = loss_eval_aver

            ckpt_path = Path("pretrained_weights") / f'ckpt' / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            ckpt_path.mkdir(parents=True, exist_ok=True)
            to_save = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            torch.save(to_save, ckpt_path / f'rew_model_ep{epoch}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VQ_VAE training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)