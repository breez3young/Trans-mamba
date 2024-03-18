# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42)
unset LD_LIBRARY_PATH
export SC2PATH="/mnt/data/optimal/zhangyang/StarCraftII"

map_name="2m_vs_1z"
env="starcraft"
seed=2


CUDA_VISIBLE_DEVICES=1 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps 50000 --mode disabled --tokenizer vq --decay 0.8 --temperature 10.0