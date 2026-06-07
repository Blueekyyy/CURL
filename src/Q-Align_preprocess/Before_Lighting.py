import os
import json
import shutil
from argparse import ArgumentParser

def main(args):
    #import pdb;pdb.set_trace()
    json_path = args.json_path
    output_dir = args.output_dir

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 读取 JSON 数据
    entries = []
    with open(json_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

    # 筛选和复制图像
    for entry in entries:
        mos = entry.get("MOS", 0)
        image_path = entry.get("image", "")
        
        if 0.3 <= mos <= 0.6 and os.path.exists(image_path):
            # 复制图像到 images_temp
            shutil.copy(image_path, output_dir)
            print(f"Copied: {image_path}")

    print("筛选和复制完成。")


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--json_path', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)