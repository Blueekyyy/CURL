export GPU="0"
#!/usr/bin/env bash

# 使用说明:
# ./batch_resize.sh 源目录 目标目录 宽度 高度
# 例如: ./batch_resize.sh ./images ./resized 800 -1

SRC_DIR="$1"
DST_DIR="$2"
WIDTH="540"
HEIGHT="960"

#if [[ -z "$SRC_DIR" || -z "$DST_DIR" || -z "$WIDTH" ]]; then
#  echo "Usage: $0 <source_dir> <dest_dir> <width> [height]"
#  exit 1
#fi

# 如果没有指定高度, 默认等比缩放 (HEIGHT=-1)
#if [[ -z "$HEIGHT" ]]; then
#  HEIGHT="-1"
#fi

# 创建目标目录
mkdir -p "$DST_DIR"

# 遍历图像
for img in "$SRC_DIR"/*.{jpg,jpeg,png,gif}; do
  # 如果文件不存在则跳过
  [[ ! -f "$img" ]] && continue

  # 提取文件名
  filename=$(basename "$img")
  
  echo "Resizing $filename ..."

  # 用 ImageMagick 进行缩放, 保持原文件名输出到目标目录
  #convert "$img" -resize "${WIDTH}x${HEIGHT}" "$DST_DIR/$filename"
  # 调用 ffmpeg 缩放
  ffmpeg -i "$img" -vf "scale=${WIDTH}:${HEIGHT}" "$DST_DIR/$filename"
done

echo "All done!"
