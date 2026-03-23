set -e 
GREEN='\e[32m'
RED='\e[31m'
RESET='\e[0m'
export OPENAI_API_KEY=""
export OPENAI_API_BASE=""
dataname="ChairDataset"  

# # ChairDataset  POPE   VDAT_Dataset   Creation_MMBench

model_name=qwen_chat

export global_seed=42
export is_use_sample="False"

result_root="result_greedy_seed${global_seed}"
export HF_ENDPOINT=https://hf-mirror.com
gpu="0,1,2,3,4,5,6,7"
use_gpu=8
   

# ################################# CHAIR 参数设置 #################################
export Chair_cache="./my_tool/CHAIR_dataset/chair_cache.pkl"
export Chair_coco_path="./data/annotations"
# ################################# VDAT 参数设置 #################################
export similarity_metric="CLIP"  # SemanticText CLIP
export VDAT_image_root="./data/random_500_from_coco/"

output_folder="./result/${result_root}/outputs_${model_name}_${dataname}"
export MMEVAL_ROOT="${output_folder}"

echo -e "$GREEN ===== Running Original Model===== $RESET"
echo -e "$GREEN ${output_folder} Ori $RESET"


# 推理
CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc-per-node=$use_gpu run.py --model ${model_name} --data $dataname  --work-dir ${output_folder} 
# 调API评估
# CUDA_VISIBLE_DEVICES=$gpu python run.py --model ${model_name} --data $dataname  --work-dir ${output_folder}  --judge gpt-4o-mini  --verbose
