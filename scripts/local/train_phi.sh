eval "$(conda shell.bash hook)"
conda activate tinyllava

NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
echo $CUDA_VISIBLE_DEVICES
PORT=29500

GLOBAL_BATCH_SIZE=256
PER_DEVICE_BATCH_SIZE=32
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / NUM_GPUS))

RUN_NAME="phi-pretrain-reproduce-fp16"
OUTPUT_DIR="output/${RUN_NAME}"

# pretrain
deepspeed --master_port ${PORT} tinyllava/train/train.py \
    --deepspeed ./configs/zero3.json \
    --config_path ./configs/training/phi-pretrain.yml \
    --data.image_folder /data/llava_data/llava/llava_pretrain/images \
    --data.data_path /data/llava_data/llava/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME} \
    --fp16 True --bf16 False