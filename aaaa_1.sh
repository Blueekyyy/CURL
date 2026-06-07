if false
then
nohup ./run2.sh > log_runa2.txt 2>&1 &
nohup ./run2_1.sh > log_runa2_1.txt 2>&1 &
nohup ./run2_2.sh > log_runa2_2.txt 2>&1 &
nohup ./run2_3.sh > log_runa2_3.txt 2>&1 &
nohup ./run2_4.sh > log_runa2_4.txt 2>&1 &

nohup ./run00.sh > log_runb00.txt 2>&1 &
nohup ./run11.sh > log_runb11.txt 2>&1 &
nohup ./run22.sh > log_runb22.txt 2>&1 &
nohup ./run33.sh > log_runb33.txt 2>&1 &
nohup ./run44.sh > log_runb44.txt 2>&1 &
fi

if true
then
(
#./run2_2_geometry_1.sh "/mnt/16T/ljy/GaussianHaircut/20251206/20251203_bomb_indoor_geometry" > log_bomb_geometry.txt 2>&1
./run2_1_texture_1.sh /mnt/16T/ljy/GaussianHaircut/20251207/20251207_curly_3_front_1_texture > log_curly_3.txt 2>&1
) &
fi