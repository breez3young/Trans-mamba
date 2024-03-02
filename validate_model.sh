# 2m_vs_1z model pth: /mnt/data/optimal/zhangyang/code/bins/results/starcraft/2m_vs_1z/run10/ckpt/model.pth
mawm_path="/mnt/data/optimal/zhangyang/code/bins/results/starcraft/2m_vs_1z/run10/ckpt/model.pth"
tokenizer="vq"

mamba_path="/mnt/data/optimal/zhangyang/code/mamba/mamba_results/starcraft/2s_vs_1sc/run1/ckpt/mamba_model.pth"

# 2m_vs_1z
map_name="2m_vs_1z"
env="starcraft"

CUDA_VISIBLE_DEVICES=1 python check_model.py --env ${env} --env_name ${map_name} --tokenizer ${tokenizer} --mawm_load_path ${mawm_path} # --mamba_load_path ${mamba_path}