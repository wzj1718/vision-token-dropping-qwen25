# # vizwiz_vqa_val  mmvet pope_adv  pope_pop  pope_random  mathvista_testmini  realworldqa
#  mirb  VisualPuzzles_direct
#  seedbench_2_plus  textvqa_val
#  scienceqa_img   mme  mmstar scienceqa  mmbench_en  mmbench_cn  mmmu_val  gqa  vizwiz_vqa_val
export CUDA_VISIBLE_DEVICES=0
# 
# drop=0时候需要改：/datas/wangzijing02/research/multimodal/Vision-Function-Layer-main/Vision-Token-Dropping/lmms_eval/models/qwen2_5_vl.py
#     inputs.pop('image_token_positions', None)  
# for DROP in {32..34}
# /datas/huggingface/Qwen2.5-VL-3B-Instruct/
# /datas/wangzijing02/research/multimodal/Vision-Function-Layer-main/merged_models/23--35/0.4base(23--35)+0.6vl
# --model_args pretrained="/datas/wangzijing02/research/multimodal/Vision-Function-Layer-main/merged_models/1base(23--35)+0.1vl--chuyi0.9,use_flash_attention_2=True" \
        
for DROP in  20 22 23
do
    echo ">>> Running drop ${DROP}..."
    python3 -m accelerate.commands.launch \
        --main_process_port 11181 \
        --num_processes=8 \
        -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained="/datas/huggingface/Qwen2.5-VL-7B-Instruct/,use_flash_attention_2=True" \
        --tasks realworldqa \
        --batch_size 1 \
        --drop ${DROP} \
        --log_samples \
        --log_samples_suffix pro-${NAME}-drop${DROP} \
        --output_path ./logs/realworldqa/Qwen2.5-VL-7B-Instruct/drop${DROP}-${NAME}

    echo ">>> Finished drop ${DROP}"
    echo "--------------------------------"
done
