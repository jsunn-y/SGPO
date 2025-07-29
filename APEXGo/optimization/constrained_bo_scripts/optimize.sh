for i in {1..5}
do
    echo "Running optimization for repeat $i"
    python apex_oracle_constrained_diverse_optimization.py \
        --task_id=sgpo \
        --divf_id=edit_dist \
        --max_n_oracle_calls=800 \
        --bsz=100 \
        --track_with_wandb=True \
        --constraint_function_ids=[] \
        --constraint_thresholds=[] \
        --constraint_types=[] \
        --wandb_entity=jasonyang12345 \
        --num_initialization_points=100 \
        --dim=256 \
        --max_string_length=400 \
        --task_specific_args=[CreiLOV] \
        --M=1 \
        --tau=1 \
        --repeat=$i \
        - run_robot - done
done
