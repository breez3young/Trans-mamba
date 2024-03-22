### 有问题的环境：3s_vs_3z() 3s_vs_3z
# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42) so_many_baneling
map_name="so_many_baneling"
env="starcraft"
seed=1


CUDA_VISIBLE_DEVICES=5 python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps 100000 --mode online --tokenizer vq --decay 0.8 --temperature 1.0 --sample_temp 20.0