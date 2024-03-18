# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42)
unset LD_LIBRARY_PATH
export SC2PATH="/mnt/data/optimal/zhangyang/StarCraftII"

map_name="2m_vs_1z"
env="starcraft"
seed=10

CUDA_VISIBLE_DEVICES=0 python train_actor.py --n_workers 1 --env ${env} --map ${map_name} --seed ${seed} \
                                                           --steps 50000 --mode disabled \
                                                           --tokenizer vq --decay 0.8 --temperature 1.0
