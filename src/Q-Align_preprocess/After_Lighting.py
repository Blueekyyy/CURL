import os
import shutil
from argparse import ArgumentParser

def main(args):
    # 设置路径
    images_dir = args.images_dir
    images_save_dir = args.images_save_dir

    #import pdb;pdb.set_trace()
    # 遍历 images_save 目录
    for filename in os.listdir(images_save_dir):
        src_path = os.path.join(images_save_dir, filename)
        dst_path = os.path.join(images_dir, filename)

        # 如果 images 中存在同名图像，就覆盖
        if os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            print(f"已覆盖：{filename}")
        else:
            print(f"未找到匹配图像：{filename}（跳过）")

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--images_dir', default='', type=str)
    parser.add_argument('--images_save_dir', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)