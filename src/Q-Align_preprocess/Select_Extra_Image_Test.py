import os
from scorer import QAlignScorer
import json
from argparse import ArgumentParser
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def score_to_text(score):
    if score < 0.1:
        return "The quality of the image is bad."
    elif score < 0.3:
        return "The quality of the image is poor."
    elif score < 0.6:
        return "The quality of the image is fair."
    elif score < 0.85:
        return "The quality of the image is good."
    else:
        return "The quality of the image is excellent."


def main(args):
    #1.读取json图像
    import pdb;pdb.set_trace()
    #with open(f"{args.json_path}", "r") as f:
        #data = json.load(f)
    with open(f"{args.json_path}", "r") as f:
        for line in f:
            data = json.loads(line)

            #2.找到最高分数类别
            logits = data["logits"]
            image_path = data["image"]
            top_quality = max(logits, key=logits.get)
            print(f"Predicted quality: {top_quality}")
            
            #3.根据类别进行光照处理并保存图像名字
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img_float = image.astype(np.float32) / 255.0 # 转换为 float32，便于处理
            
            if top_quality in ["low", "poor", "bad"]:
                # 低光增强（增强亮度）
                #enhanced = np.clip(img_float * 1.5, 0, 1)
                print("Applied low-light enhancement.")
            elif top_quality in ["high", "excellent"]:
                # 高光抑制（降低亮度）
                #enhanced = np.clip(img_float * 0.7, 0, 1)
                print("Applied highlight suppression.")
            else:
                enhanced = img_float  # 不处理
                print("No enhancement applied.")
            
            #4.筛选出128张图像

            #5.放入images_128文件夹
            enhanced_uint8 = (enhanced * 255).astype(np.uint8)
            enhanced_bgr = cv2.cvtColor(enhanced_uint8, cv2.COLOR_RGB2BGR)

            #cv2.imwrite("enhanced_output.png", enhanced_bgr) # 保存新图像


    pass


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--json_path', default='', type=str)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()
   
    main(args)
