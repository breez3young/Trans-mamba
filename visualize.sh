unset LD_LIBRARY_PATH
export SC2PATH="/mnt/data/optimal/zhangyang/StarCraftII"

# 2m_vs_1z model pth: /mnt/data/optimal/zhangyang/code/bins/results/starcraft/2m_vs_1z/run10/ckpt/model.pth
# 2s_vs_1sc: /mnt/data/optimal/zhangyang/code/bins/results/starcraft/2s_vs_1sc/run14/ckpt/model.pth
# mawm_path="/mnt/data/optimal/zhangyang/code/bins/results/starcraft/2m_vs_1z_vq/run22/ckpt/model_10Ksteps.pth"
mawm_path="/mnt/data/optimal/zhangyang/code/bins/results/starcraft/2m_vs_1z_vq/run16/ckpt/model_13Ksteps.pth"
tokenizer="vq"

# 2m_vs_1z model pth (seed 123): /mnt/data/optimal/zhangyang/code/mamba/mamba_results/starcraft/2m_vs_1z/run2/ckpt/mamba_model.pth
# 2s_vs_1sc: /mnt/data/optimal/zhangyang/code/mamba/mamba_results/starcraft/2s_vs_1sc/run1/ckpt/mamba_model.pth
mamba_path="/mnt/data/optimal/zhangyang/code/mamba/mamba_results/starcraft/2s_vs_1sc/run1/ckpt/mamba_model.pth"

# 2m_vs_1z
map_name="2m_vs_1z"
env="starcraft"

python visualize.py --env ${env} --map_name ${map_name} --tokenizer ${tokenizer} \
                    --model_path ${mawm_path} \
                    --eval_episodes 10