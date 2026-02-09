# Launch LIBERO-Spatial evals
export TORCHDYNAMO_DISABLE=1
export CUDA_VISIBLE_DEVICES=0

# RUN AFTER VENV-LIBERO IS ACTIVATED

# python examples/libero/main.py \
#   --args.task_suite_name libero_spatial \

# python examples/libero/main.py \
#   --args.task_suite_name libero_object \

# python examples/libero/main.py \
# --args.task_suite_name libero_goal \

python examples/libero/main.py \
--args.task_suite_name libero_10 \

# if you want to eval on LIBERO-Object/Goal/10,
# just modify the settings  `--args.task_suite_name` 