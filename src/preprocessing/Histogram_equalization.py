import cv2
import glob
import os
from argparse import ArgumentParser


def main(args):
    # 读取 images_2 目录下的所有彩色图像
    image_paths = glob.glob(f'{args.data_path}/images_2/*.jpg') \
                 + glob.glob(f'{args.data_path}/images_2/*.png') \
                 + glob.glob(f'{args.data_path}/images_2/*.jpeg')

    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for path in image_paths:
        # 读取彩色图像
        img = cv2.imread(path)

        if img is None:
            print(f"无法读取图像：{path}")
            continue

        # 转换到 YCrCb 空间
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # 拆分通道
        y, cr, cb = cv2.split(ycrcb)

        # 对亮度通道做 CLAHE 直方图均衡化
        y_clahe = clahe.apply(y)

        # 合并通道
        ycrcb_clahe = cv2.merge((y_clahe, cr, cb))

        # 转回 BGR 空间
        img_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

        # 覆盖保存
        cv2.imwrite(path, img_clahe)
        print(f"已处理并覆盖：{path}")

    print("批量 CLAHE 直方图均衡化完成！")


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--data_path', default='', type=str)
    args = parser.parse_args()
    main(args)
