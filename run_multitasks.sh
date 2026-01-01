
#  scienceqa_img   mme  mmstar scienceqa  mmbench_en  mmbench_cn  mmmu_val
export CUDA_VISIBLE_DEVICES=5
  
#  scienceqa_img   mme  mmstar scienceqa  mmbench_en  mmbench_cn  mmmu_val  gqa  seedbench_2_plus  textvqa_val
#   - pope_adv   - pope_pop  - pope_random
TASKS=("mmmu_val")   # 你想跑的任务列表
DROPS=({4..8})


for TASK in "${TASKS[@]}"
do
    echo ">>> Running TASK ${TASK}..."

    for DROP in "${DROPS[@]}"
    do
        echo ">>>   Running DROP ${DROP}..."

        python3 -m accelerate.commands.launch \
            --main_process_port 13077 \
            --num_processes=8 \
            -m lmms_eval \
            --model qwen2_5_vl \
            --model_args pretrained="/datas/wangzijing02/research/multimodal/Vision-Function-Layer-main/merged_models_7B/mmmu/merge_21--27+0.4base+0.8vl,use_flash_attention_2=True" \
            --tasks ${TASK} \
            --batch_size 1 \
            --drop ${DROP} \
            --log_samples \
            --log_samples_suffix pro-${NAME}-${TASK}-drop${DROP} \
            --output_path ./logs/${TASK}/Qwen2.5-VL-7B/${TASK}-drop${DROP}-${NAME}

        echo ">>>   Finished DROP ${DROP}"
        echo "--------------------------------"
    done

    echo ">>> Finished TASK ${TASK}"
    echo "================================"
done



