# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42)
unset LD_LIBRARY_PATH
export SC2PATH="/mnt/data/optimal/zhangyang/StarCraftII"

map_name="2m_vs_1z"
env="starcraft"

CUDA_VISIBLE_DEVICES=0 python pretrain_wm.py --env ${env} --map ${map_name} --num_steps 50000 \
                                             --mode disabled \
                                             --tokenizer vq --decay 0.8 \
                                             --epochs 500
