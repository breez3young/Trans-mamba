# map: 3m  so_many_baneling
map_name="so_many_baneling"
env="starcraft"
seed=2


CUDA_VISIBLE_DEVICES=6 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --mode online