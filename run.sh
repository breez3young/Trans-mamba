# train reward estimator (using offline dataset)
python train_reward_estimator.py --epochs 200 --batch_size 128 --mode disabled


# MAIN EXP
## train world model + online learning
CUDA_VISIBLE_DEVICES=4 python train.py --n_workers 1 --env starcraft --env_name so_many_baneling --steps 50000 --mode online --tokenizer vq --decay 0.8 --temperature 10.0 --seed 1
CUDA_VISIBLE_DEVICES=5 python train.py --n_workers 1 --env starcraft --env_name so_many_baneling --steps 50000 --mode online --tokenizer vq --decay 0.8 --temperature 10.0 --seed 2
CUDA_VISIBLE_DEVICES=6 python train.py --n_workers 1 --env starcraft --env_name so_many_baneling --steps 50000 --mode online --tokenizer vq --decay 0.8 --temperature 10.0 --seed 3
CUDA_VISIBLE_DEVICES=7 python train.py --n_workers 1 --env starcraft --env_name so_many_baneling --steps 50000 --mode online --tokenizer vq --decay 0.8 --temperature 10.0 --seed 4