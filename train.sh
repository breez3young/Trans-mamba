# map: 3m  so_many_baneling 2c_vs_64zg 3s5z_vs_3s6z 6h_vs_8z 8m_vs_9m 2s_vs_1sc 2m_vs_1z
map_name="so_many_baneling"
env="starcraft"
seed=2


CUDA_VISIBLE_DEVICES=5 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps 200000 --mode online