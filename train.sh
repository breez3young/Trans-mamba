# map: 2s_vs_1sc(17) 2m_vs_1z(16) 3s_vs_4z(42) so_many_baneling MMM 2s3z
map_name="2s3z"
env="starcraft"
seed=2
cuda_device=0

# --ce_for_av
CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --n_workers 1 --env ${env} --env_name ${map_name} --seed ${seed} --steps 100000 --mode online --tokenizer vq --decay 0.8 \
                                                    --temperature 2.0 --sample_temp inf --ce_for_av