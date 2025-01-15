export HF_HOME=/home/ps/wk/cache_model/huggingface_model
export TORCH_HOME=/home/ps/wk/cache_model/torch_model

python3 deploy.py \
        --ckpt_dir /media/ps/wk/benchmark_results/act/franka_3rgb_pick_plate_from_plate_rack/ckpt \
        --exp_type franka_3rgb \
        