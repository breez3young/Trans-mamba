# map: 3m  so_many_baneling 2c_vs_64zg 3s5z_vs_3s6z 6h_vs_8z
map_name="so_many_baneling"
env="starcraft"
seed=3


CUDA_VISIBLE_DEVICES=7 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps 100000 # --mode online