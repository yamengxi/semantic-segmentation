srun -p pat_mars1 \
    --job-name=MoV3s \
    --gres=gpu:8 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=5 \
    --kill-on-bad-exit=1 \
    python -u -m runx.runx scripts/train_mobilev3small.yml -i & # > mobilev3small_log.txt &
