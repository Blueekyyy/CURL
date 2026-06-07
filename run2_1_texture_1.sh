export GPU="1"
export CAMERA="PINHOLE"
export EXP_NAME_1="stage1"
export EXP_NAME_2="stage2"
export EXP_NAME_3="stage3"

export PROJECT_DIR="/mnt/data/ljy/Li/GaussianHaircut"
export BLENDER_DIR="/mnt/data/ljy/Li/blender-3.6.19-linux-x64/blender"
#export DATA_PATH="/mnt/16T/ljy/GaussianHaircut/20251203_1GH_LowLight/20251203_1GH_bomb_texture"
export DATA_PATH="$1"
echo "DATA_PATH is set to: $DATA_PATH"

eval "$(conda shell.bash hook)"

export EXP_PATH_1=$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1

if false
then
    #从视频中抽取帧
    # Arrange raw images into a 3D Gaussian Splatting format
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python preprocess_raw_images_copy_Q-Align.py \
        --data_path $DATA_PATH
fi

# ！！！！！！！！！下面两步不执行，低光 & 背光数据集/正常光照数据集人为判断是否后续全部用/全部不用HVI光照处理！！！！！！！！！！！
if false
then
    #得到视频帧的json文件和基础分数input.json
    conda deactivate && conda activate Q-Align
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python Image_score_Test.py \
        --data_path $DATA_PATH --mode "input" \
        --subfolder "input"
fi

if false
then
    #得到微调后的词条分数转为MOS生成aes-input.json，同时筛掉MOS分数低于0.3的帧
    conda deactivate && conda activate Q-Align
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python iqa_eval_lora_split_copy.py \
        --model-path /mnt/data/ljy/p/IQA/Q-Align/q-align-gh-lora-4 \
        --model-base q-future/one-align \
        --data_path $DATA_PATH \
        --json_path $DATA_PATH/input.json \
        --save_path $DATA_PATH/aes-input-1.json
fi
# ！！！！！！！！！上面两步不执行，低光 & 背光数据集/正常光照数据集人为判断是否后续全部用/全部不用HVI光照处理！！！！！！！！！！！

# 弃用！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
if false
then
    #根据MOS值排序得到高分300张图像存入inputs目录
    #修改：根据MOS值每隔len(input)/300选最高分一帧图像存入inputs目录
    #总图像质量少于300时提示“数据集质量不过关”
    conda deactivate && conda activate Q-Align
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python Select_Raw_Image_Test.py \
        --data_path $DATA_PATH \
        --json_path $DATA_PATH/aes-input-1.json
fi
# 弃用！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

if true
then
    #从input目录运行colmap
    # Run COLMAP reconstruction and undistort the images and cameras
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python convert_copy_texture.py -s $DATA_PATH \
        --camera $CAMERA --max_size 1024
fi

if true
then
    #从images目录计算mask
    # Run Matte-Anything
    conda deactivate && conda activate matte_anything
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_masks.py \
        --data_path $DATA_PATH --image_format png --max_size 2048
fi

# !!!!!!!!!!下面这步替换成低光、背光数据集直接images目录复制到images_temp，正常光照数据集则不复制，空目录!!!!!!!!!!!!
if false
then
    #根据aes-input-1.json文件，MOS分数大于0.6不做处理，分数在0.0-0.6之间进行低光增强
    #将需要被处理的图像存到images_temp目录，进行光照处理得到images_save目录，之后再覆盖原来images存的原图像，最后用images目录做图像缩小分辨率
    # ====== new ======
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python Before_Lighting.py \
            --data_path $DATA_PATH \
            --json_path $DATA_PATH/aes-input-1.json \
            --output_dir $DATA_PATH/images_temp
fi

if true
then
    cp -r $DATA_PATH/images $DATA_PATH/images_temp
fi

if true
then
    #HVI低光增强（不必要放到mask之前）
    #换位置：为防止筛选处理后有些图像名存入了但图像被舍弃，将光照处理和filter_extra_images_copy换一下位置
    conda deactivate && conda activate CIDNet
    cd $PROJECT_DIR/ext/HVI-CIDNet
    CUDA_VISIBLE_DEVICES="$GPU" ./Process_img.sh "$DATA_PATH/images_temp" "$DATA_PATH/images_proc"
fi

#=== New ===
if true
then
    #为后面的第一次的resize提供images_save目录
    cp -rv $DATA_PATH/images $DATA_PATH/images_save
fi

if true
then
    #从images_save目录读入图像名称，只保留images_2的resize，不保留mask
    # Resize images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python resize_images_copy_copy.py --data_path $DATA_PATH
fi

if true
then
    #保留images_2_save目录，作为后序step3.2解耦几何颜色训练的读入图像
    mv $DATA_PATH/images_2 $DATA_PATH/images_2_save
fi

if true
then
    #将images_proc目录里存的图像覆盖images里同名图像
    # ====== new ======
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python After_Lighting.py \
            --images_dir $DATA_PATH/images \
            --images_save_dir $DATA_PATH/images_proc
fi

if true
then
    #对images目录重新打一次分
    #得到images.json
    # ====== new ======
    conda deactivate && conda activate Q-Align
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python Image_score_Test.py \
        --data_path $DATA_PATH --mode "images" \
        --subfolder "images"
fi

if true
then
    #算第二次分数
    # ====== new ======
    conda deactivate && conda activate Q-Align
    cd $PROJECT_DIR/src/Q-Align_preprocess
    CUDA_VISIBLE_DEVICES="$GPU" python iqa_eval_lora_split_copy_texture.py \
        --model-path /mnt/data/ljy/p/IQA/Q-Align/q-align-gh-lora-4 \
        --model-base q-future/one-align \
        --data_path $DATA_PATH \
        --json_path $DATA_PATH/images.json \
        --save_path $DATA_PATH/aes-input-2.json
fi

if true
then
    #根据第2次分数筛选128张图像
    #筛选出128张图像名称存入iqa_filtered_names.pkl，将图像分成128组bin，每组bin取分数最高值
    #在这一步获得质量分数最差的一帧
    # Filter images using their IQA scores
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python filter_extra_images_copy.py \
        --data_path $DATA_PATH --max_imgs 128
fi

if true
then
    #从images目录或从iqa_filtered_names.pkl读入图像名称
    # Resize images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python resize_images_copy.py --data_path $DATA_PATH
fi

if true
then
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python Histogram_equalization.py --data_path $DATA_PATH
fi


if true
then
# Calculate orientation maps
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_orientation_maps.py \
    --img_path $DATA_PATH/images_2 \
    --mask_path $DATA_PATH/masks_2/hair \
    --orient_dir $DATA_PATH/orientations_2/angles \
    --conf_dir $DATA_PATH/orientations_2/vars \
    --filtered_img_dir $DATA_PATH/orientations_2/filtered_imgs \
    --vis_img_dir $DATA_PATH/orientations_2/vis_imgs

# Run OpenPose
conda deactivate && cd $PROJECT_DIR/ext/openpose
mkdir $DATA_PATH/openpose
CUDA_VISIBLE_DEVICES="$GPU" ./build/examples/openpose/openpose.bin \
    --image_dir $DATA_PATH/images_4 \
    --scale_number 4 --scale_gap 0.25 --face --hand --display 0 \
    --write_json $DATA_PATH/openpose/json \
    --write_images $DATA_PATH/openpose/images --write_images_format jpg

# Run Face-Alignment
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python calc_face_alignment.py \
    --data_path $DATA_PATH --image_dir "images_4"

# Run PIXIE
conda deactivate && conda activate pixie-env
cd $PROJECT_DIR/ext/PIXIE
CUDA_VISIBLE_DEVICES="$GPU" python demos/demo_fit_face.py \
    -i $DATA_PATH/images_4 -s $DATA_PATH/pixie \
    --saveParam True --lightTex False --useTex False \
    --rasterizer_type pytorch3d

# Merge all PIXIE predictions in a single file
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python merge_smplx_predictions.py \
    --data_path $DATA_PATH

# Convert COLMAP cameras to txt
conda deactivate && conda activate gaussian_splatting_hair
mkdir $DATA_PATH/sparse_txt
CUDA_VISIBLE_DEVICES="$GPU" colmap model_converter \
    --input_path $DATA_PATH/sparse/0  \
    --output_path $DATA_PATH/sparse_txt --output_type TXT

# Convert COLMAP cameras to H3DS format
conda deactivate && conda activate gaussian_splatting_hair
cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python colmap_parsing.py \
    --path_to_scene $DATA_PATH
fi

if false
then
    conda deactivate && conda activate vllm
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python Complex_Score.py \
        --data_path $DATA_PATH
fi

if false
then
    conda deactivate && conda activate vllm
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python Light_Score.py \
        --data_path $DATA_PATH
fi