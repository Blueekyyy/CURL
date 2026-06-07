export GPU="1"
export CAMERA="PINHOLE"
export EXP_NAME_1="stage1"
export EXP_NAME_2="stage2"
export EXP_NAME_3="stage3"

export PROJECT_DIR="/mnt/data/ljy/Li/GaussianHaircut"
export BLENDER_DIR="/mnt/data/ljy/Li/blender-3.6.19-linux-x64/blender"
#export DATA_PATH="/mnt/16T/ljy/GaussianHaircut/20251127_indoor_1"
#export DATA_PATH="../input_exp"
export DATA_PATH="$1"
echo "DATA_PATH is set to: $DATA_PATH"
export POINT_CLOUD_PATH_HEAD="$2"
export GS_OUTPUT="$3"

eval "$(conda shell.bash hook)"

export EXP_PATH_1=$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1

if false
then
conda deactivate && conda activate gaussian_splatting
cd /mnt/data/ljy/Li/gaussian-splatting
cd Depth-Anything-V2
CUDA_VISIBLE_DEVICES="$GPU" python run.py --encoder vitl --pred-only --grayscale \
    --img-path $DATA_PATH/images_2 \
    --outdir $DATA_PATH/depth
cd ..

CUDA_VISIBLE_DEVICES="$GPU" python utils/make_depth_scale.py --base_dir $DATA_PATH \
    --depths_dir $DATA_PATH/depth

CUDA_VISIBLE_DEVICES="$GPU" python train.py -s $DATA_PATH \
    -m $GS_OUTPUT \
    -d depth \
    --antialiasing --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp


fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_perceptual_loss_vgg.py \
        --image_folder $DATA_PATH/images_2 \
        --save_folder1 $DATA_PATH/vgg_features_pct_pth_1 \
        --save_folder2 $DATA_PATH/vgg_features_pct_png_1 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_edge_loss_canny.py \
        --image_folder $DATA_PATH/images_2 \
        --save_folder1 $DATA_PATH/edge_canny_pth_1 \
        --save_folder2 $DATA_PATH/edge_canny_png_1 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_edge_loss.py \
        --image_folder $DATA_PATH/images_2 \
        --save_folder1 $DATA_PATH/edge_canny_pth_1 \
        --save_folder2 $DATA_PATH/edge_canny_png_1 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run 3D Gaussian Splatting reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
        -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
        --trainable_cameras --trainable_intrinsics --use_barf \
        --lambda_dorient 0.1 \
        --pointcloud_path_head "$POINT_CLOUD_PATH_HEAD" \
        --is_densify
fi

if false
then
    # Run 3D Gaussian Splatting reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
        -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
        --trainable_cameras --trainable_intrinsics --use_barf \
        --lambda_dorient 0.1 \
        --pointcloud_path_head "$POINT_CLOUD_PATH_HEAD" \
        --start_checkpoint $EXP_PATH_1/checkpoints/10000.pth
fi

if false
then
    # Run FLAME mesh fitting
    conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/ext/NeuralHaircut/src/multiview_optimization

    CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1.conf \
        --batch_size 1 --train_rotation True --fixed_images True \
        --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_1 \
        --data_path $DATA_PATH \
        --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

    CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1.conf \
        --batch_size 4 --train_rotation True --fixed_images True \
        --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_2 \
        --checkpoint_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_1/opt_params_final \
        --data_path $DATA_PATH \
        --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

    CUDA_VISIBLE_DEVICES="$GPU" python fit.py --conf confs/train_person_1_.conf \
        --batch_size 32 --train_rotation True --train_shape True \
        --save_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3 \
        --checkpoint_path $DATA_PATH/flame_fitting/$EXP_NAME_1/stage_2/opt_params_final \
        --data_path $DATA_PATH \
        --fitted_camera_path $EXP_PATH_1/cameras/30000_matrices.pkl

    # Crop the reconstructed scene
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python scale_scene_into_sphere.py \
        --path_to_data $DATA_PATH \
        -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000

    # Remove hair Gaussians that intersect with the FLAME head mesh  # UserWarning: No mtl file provided
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python filter_flame_intersections.py \
        --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
        -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iter 30000 \
        --project_dir $PROJECT_DIR/ext/NeuralHaircut
fi

if false
then
    # Run rendering for training views   
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python render_gaussians.py \
        -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" \
        --skip_test --scene_suffix "_cropped" --iteration 30000 \
        --trainable_cameras --trainable_intrinsics --use_barf \
        --pointcloud_path_head "$EXP_PATH_1/point_cloud_cropped/iteration_30000/raw_point_cloud.ply"
        #--pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply"
        #--pointcloud_path_head None
fi

if false
then
    # Get FLAME mesh scalp maps
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing   # 获取FLAME mesh 头皮图
    CUDA_VISIBLE_DEVICES="$GPU" python extract_non_visible_head_scalp.py \
        --project_dir $PROJECT_DIR/ext/NeuralHaircut --data_dir $DATA_PATH \
        --flame_mesh_dir $DATA_PATH/flame_fitting/$EXP_NAME_1 \
        --cams_path $DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1/cameras/30000_matrices.pkl \
        -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1"
fi

if false
then
    # Run Perceptual loss from rendering images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_perceptual_loss_vgg.py \
        --image_folder $EXP_PATH_1/train_cropped/ours_30000/renders \
        --save_folder1 $DATA_PATH/vgg_features_pct_pth_2 \
        --save_folder2 $DATA_PATH/vgg_features_pct_png_2 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then

    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_edge_loss_canny.py \
        --image_folder $EXP_PATH_1/train_cropped/ours_30000/renders \
        --save_folder1 $DATA_PATH/edge_canny_pth_2 \
        --save_folder2 $DATA_PATH/edge_canny_png_2 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_edge_loss.py \
        --image_folder $EXP_PATH_1/train_cropped/ours_30000/renders \
        --save_folder1 $DATA_PATH/edge_canny_pth_2 \
        --save_folder2 $DATA_PATH/edge_canny_png_2 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run latent hair strands reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_latent_strands.py \
        -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
        --model_path_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2" \
        --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
        --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
        --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
        --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
        --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
        --trainable_cameras --trainable_intrinsics --use_barf \
        --iterations 20000 --port "800$GPU"
        #--pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
        #--iterations 2000
fi

if false
then
    # step2.2
    # Run latent hair strands reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_latent_strands.py \
        -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
        --model_path_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2" \
        --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
        --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
        --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
        --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
        --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
        --trainable_cameras --trainable_intrinsics --use_barf \
        --iterations 20000 --port "800$GPU" \
        --is_step2_2 \
        --checkpoint_hair_ljy "$DATA_PATH/strands_reconstruction/stage2/checkpoints/20000.pth"
fi

if false
then
    # Run hair strands reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_strands.py \
        -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
        --model_path_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
        --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
        --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
        --start_checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
        --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
        --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
        --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
        --position_lr_init 0.0000016 --position_lr_max_steps 10000 \
        --trainable_cameras --trainable_intrinsics --use_barf \
        --iterations 10000 --port "800$GPU"
        #--pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
        #--start_checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/2000.pth" \
        #--iterations 1000
fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_perceptual_loss_vgg.py \
        --image_folder $DATA_PATH/images_2_save \
        --save_folder1 $DATA_PATH/vgg_features_pct_pth_light_3 \
        --save_folder2 $DATA_PATH/vgg_features_pct_png_light_3 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_edge_loss_canny.py \
        --image_folder $DATA_PATH/images_2_save \
        --save_folder1 $DATA_PATH/edge_canny_pth_light_3 \
        --save_folder2 $DATA_PATH/edge_canny_png_light_3 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # Run Perceptual loss from origin images
    conda deactivate && conda activate gaussian_splatting_hair
    cd $PROJECT_DIR/src/preprocessing
    CUDA_VISIBLE_DEVICES="$GPU" python calc_edge_loss.py \
        --image_folder $DATA_PATH/images_2_save \
        --save_folder1 $DATA_PATH/edge_canny_pth_light_3 \
        --save_folder2 $DATA_PATH/edge_canny_png_light_3 \
        --mask_folder $DATA_PATH/masks_2/body
fi

if false
then
    # step3.2
    # Run hair strands reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_strands.py \
        -s $DATA_PATH -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" -r 1 \
        --model_path_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
        --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
        --pointcloud_path_head "$EXP_PATH_1/point_cloud_filtered/iteration_30000/raw_point_cloud.ply" \
        --start_checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
        --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
        --lambda_dmask 0.1 --lambda_dorient 0.1 --lambda_dsds 0.01 \
        --load_synthetic_rgba --load_synthetic_geom --binarize_masks --iteration_data 30000 \
        --position_lr_init 0.0000016 --position_lr_max_steps 10000 \
        --iterations 10000 --port "800$GPU" \
        --is_feature_dc_rest \
        --start_checkpoint_curve "$DATA_PATH/curves_reconstruction/stage3/checkpoints/10000.pth" \
        --load_origin_rgba_geom \
        --sobel_loss
fi

if false
then
    # step 1.3: optimize body & cloth's color
    # Run 3D Gaussian Splatting reconstruction
    conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
    CUDA_VISIBLE_DEVICES="$GPU" python train_gaussians.py \
        -s $DATA_PATH -m "$EXP_PATH_1" -r 1 --port "888$GPU" \
        --trainable_cameras --trainable_intrinsics \
        --lambda_dorient 0.1 \
        --pointcloud_path_head "$POINT_CLOUD_PATH_HEAD" \
        --start_checkpoint $EXP_PATH_1/checkpoints/30000.pth \
        --is_feature_dc_rest \
        --load_origin_rgba_geom \
        --sobel_loss
fi

if false
then
# Export the resulting strands as pkl and ply
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/preprocessing
CUDA_VISIBLE_DEVICES="$GPU" python export_curves.py \
    --data_dir $DATA_PATH --model_name $EXP_NAME_3 --iter 10000 \
    --flame_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/stage_3/mesh_final.obj" \
    --scalp_mesh_path "$DATA_PATH/flame_fitting/$EXP_NAME_1/scalp_data/scalp.obj" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml"
    #--iter 1000
fi

if false
then
# Render the visualizations
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
CUDA_VISIBLE_DEVICES="$GPU" python render_video.py \
    --blender_path "$BLENDER_DIR" --input_path "$DATA_PATH" \
    --exp_name_1 "$EXP_NAME_1" --exp_name_3 "$EXP_NAME_3"
fi

if true
then
# Render the strands
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src
CUDA_VISIBLE_DEVICES="$GPU" python render_strands.py \
    -s $DATA_PATH --data_dir "$DATA_PATH" --data_device 'cpu' --skip_test \
    -m "$DATA_PATH/3d_gaussian_splatting/$EXP_NAME_1" --iteration 30000 \
    --flame_mesh_dir "$DATA_PATH/flame_fitting/$EXP_NAME_1" \
    --model_hair_path "$DATA_PATH/curves_reconstruction/$EXP_NAME_3" \
    --hair_conf_path "$PROJECT_DIR/src/arguments/hair_strands_textured.yaml" \
    --checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/20000.pth" \
    --checkpoint_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3/checkpoints/10000.pth" \
    --pointcloud_path_head "$EXP_PATH_1/point_cloud/iteration_30000/raw_point_cloud.ply" \
    --interpolate_cameras
    #--checkpoint_hair "$DATA_PATH/strands_reconstruction/$EXP_NAME_2/checkpoints/2000.pth" \
    #--checkpoint_curves "$DATA_PATH/curves_reconstruction/$EXP_NAME_3/checkpoints/1000.pth" \
fi

if true
then
# Make the video
conda activate gaussian_splatting_hair && cd $PROJECT_DIR/src/postprocessing
CUDA_VISIBLE_DEVICES="$GPU" python concat_video.py \
    --input_path "$DATA_PATH" --exp_name_3 "$EXP_NAME_3"
fi