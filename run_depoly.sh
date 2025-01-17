export HF_HOME=/home/ps/wk/cache_model/huggingface_model
export TORCH_HOME=/home/ps/wk/cache_model/torch_model

python3 deploy.py \
        --model_dir /media/users/will/outputs/2025.01.14/17.42.54_equi_diff_close_the_pot/ \
        --ckpt_name checkpoints/latest.ckpt
        
        