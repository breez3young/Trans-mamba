mawm_path="/mnt/data/optimal/zhangyang/code/bins/results/starcraft/2s_vs_1sc/run14/ckpt/model.pth"
tokenizer="vq"

map_name="2s_vs_1sc"
env="starcraft"

python check_model.py --env ${env} --env_name ${map_name} --tokenizer ${tokenizer} --mawm_load_path ${mawm_path}