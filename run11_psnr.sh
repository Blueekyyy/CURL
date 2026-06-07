export GPU="0"

export PROJECT_DIR="/mnt/data/ljy/Li/GaussianHaircut"
export DATA_PATH="$1"
echo "DATA_PATH is set to: $DATA_PATH"

eval "$(conda shell.bash hook)"

if true
then
# Render the visualizations
conda deactivate && conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
CUDA_VISIBLE_DEVICES="$GPU" python evaluate_metrics.py \
    --raw_dir "$DATA_PATH/curves_reconstruction/stage3/raw_frames_resized" \
    --mask_dir "$DATA_PATH/masks_hair_resized" \
    --rend_dir "$DATA_PATH/curves_reconstruction/stage3/train/ours_30000/renders" \
    --out_txt "$DATA_PATH/texture_metrics.txt"
fi