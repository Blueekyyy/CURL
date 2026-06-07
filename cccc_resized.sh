if true
then
(
./run2_resized.sh "/mnt/16T/ljy/GaussianHaircut/20251208/20260416_curly_0_-2.0_ablation_5/curves_reconstruction/stage3/raw_frames" \
    /mnt/16T/ljy/GaussianHaircut/20251208/20260416_curly_0_-2.0_ablation_5/curves_reconstruction/stage3/raw_frames_resized
./run2_resized.sh "/mnt/16T/ljy/GaussianHaircut/20251208/20260416_curly_0_-2.0_ablation_5/masks/hair" \
    /mnt/16T/ljy/GaussianHaircut/20251208/20260416_curly_0_-2.0_ablation_5/masks_hair_resized
#./run2_resized.sh /mnt/16T/ljy/GaussianHaircut/20251212_NH/20251213_igor/input \
#    /mnt/16T/ljy/GaussianHaircut/20251212_NH/20251213_igor/input_resized
#./run2_resized.sh /mnt/16T/ljy/GaussianHaircut/20251212_NH/20251213_nastya/input \
#    /mnt/16T/ljy/GaussianHaircut/20251212_NH/20251213_nastya/input_resized
) &
fi