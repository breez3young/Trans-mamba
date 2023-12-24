# map: 3m  so_many_baneling
map_name="so_many_baneling"
env="starcraft"
seed=4


CUDA_VISIBLE_DEVICES=7 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --mode online


sh /mnt/data/optimal/zhangyang/mawm_8m9m.sh

sh /mnt/data/optimal/zhangyang/mawm_train.sh