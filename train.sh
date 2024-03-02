# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42)
map_name="2m_vs_1z"
env="starcraft"
seed=2


CUDA_VISIBLE_DEVICES=0 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps 50000 --mode disabled --tokenizer fsq