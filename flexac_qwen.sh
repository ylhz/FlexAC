set -e 

GREEN='\e[32m'
RED='\e[31m'
RESET='\e[0m'
export HF_ENDPOINT=https://hf-mirror.com

export OPENAI_API_KEY=""
export OPENAI_API_BASE=""

dataset_list=("VDAT_Dataset")

# # ChairDataset  POPE   VDAT_Dataset   Creation_MMBench


gpu="0,1,2,3,4,5,6,7"
use_gpu=8
export global_seed=42

# greedy
export is_use_sample="False"
result_root="result_greedy_seed${global_seed}"


model_list=(qwen_chat) 
export Chair_cache="./my_tool/CHAIR_dataset/chair_cache.pkl"
export Chair_coco_path="./data/annotations"
export similarity_metric="CLIP" 
export VDAT_image_root="./data/random_500_from_coco/"



for model_name in "${model_list[@]}"; do


    top_image_num=(50)

    for num in "${top_image_num[@]}"; do
        export FLEXAC_TOPN=${num}

        export FLEXAC_DISTANCE_TYPE="cosine" # cosine

        if [ "$model_name" == "qwen_chat" ]; then
            control_ROOT="./general_vector/feature"  
            export RUN_NAME_EVAL_CREATION_FLEXAC_BENCH_PATH="${control_ROOT}"
        else
            echo -e "$RED 当前模型对应的向量还未生成 $RESET"
            exit 1
        fi

        for dataname in "${dataset_list[@]}"; do
            echo -e "$GREEN ===== Processing dataset: $dataname ===== $RESET"

            ################################# FlexAC-P #################################
            output_folder="./result/${result_root}/outputs_${model_name}_flexac_p_${FLEXAC_DISTANCE_TYPE}_top${FLEXAC_TOPN}_seed${global_seed}"
            export MMEVAL_ROOT="${output_folder}"
            export FLEXAC_CONTROL_FACTOR=-1          

            echo -e "$GREEN ===== Running FlexAC-P ===== $RESET"
            echo -e "$GREEN ${output_folder} P $RESET"

            CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc-per-node=$use_gpu run.py --model ${model_name}_flexac_top --data $dataname --work-dir ${output_folder}  
            # CUDA_VISIBLE_DEVICES=$gpu python run.py --model ${model_name}_flexac_top --data $dataname --work-dir ${output_folder} --judge gpt-4o-mini  --verbose

            ################################# FlexAC-C #################################
            # output_folder="${result_root}/outputs_${model_name}_flexac_c_${FLEXAC_DISTANCE_TYPE}_top${FLEXAC_TOPN}_seed${global_seed}"
            # export MMEVAL_ROOT="${output_folder}"
            # export FLEXAC_CONTROL_FACTOR=1            

            # echo -e "$GREEN ===== Running FlexAC-C ===== $RESET"
            # echo -e "$GREEN ${output_folder} C $RESET"

            # CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc-per-node=$use_gpu run.py --model ${model_name}_flexac_top --data $dataname --work-dir ${output_folder} 
        done
    done
done




