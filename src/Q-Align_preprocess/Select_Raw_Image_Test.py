import json
import os
import shutil
from argparse import ArgumentParser
from math import ceil

def main(args):
    #import pdb;pdb.set_trace()
    input_json_path = args.json_path
    output_dir = os.path.join(args.data_path,"inputs")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 读取每一行 JSON 对象
    data = []
    with open(input_json_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            data.append(obj)
    
    total = len(data)
    #segment = total // 300  # 每段的长度（向下取整）
    target_num = 300
    num_groups = min(target_num, total)  # 不超过总数
    segment = total / num_groups  # 使用浮点除法确保尽量平均
    
    selected_items = []

    #当总图像小于300张时提示“数据集质量不过关”
    #if total < 300:
    #    raise ValueError("The quality of the dataset is not up to standard.")

    #import pdb;pdb.set_trace()
    # Step 2: 根据 logits['excellent'] 排序（从高到低）
    #data_sorted = sorted(data, key=lambda x: x["MOS"], reverse=True)
    #data_sorted = sorted(data, key=lambda x: x["MOS"], reverse=True)

    # Step 3: 取前 300 项，复制图片文件
    #for item in data_sorted[:300]:
    #    src_img_path = item["image"]
    #    dst_img_path = os.path.join(output_dir, os.path.basename(src_img_path))
    #    shutil.copy(src_img_path, dst_img_path)
    
    #import pdb;pdb.set_trace()
    # Step 2: 分组并挑选每组 MOS 最高的项
    for i in range(num_groups):
        start_idx = int(i * segment)
        end_idx = int((i + 1) * segment)
        group = data[start_idx:end_idx]
        if not group:
            continue
        best_item = max(group, key=lambda x: x["MOS"])
        selected_items.append(best_item)

    # Step 3: 复制图片
    for item in selected_items:
        src_img_path = item["image"]
        dst_img_path = os.path.join(output_dir, os.path.basename(src_img_path))
        shutil.copy(src_img_path, dst_img_path)
    
    #for i in range(0, total, segment):
    #    group = data[i:i + segment]
    #    if not group:
    #        continue
    #    best_item = max(group, key=lambda x: x["MOS"])
    #    src_img_path = best_item["image"]
    #    dst_img_path = os.path.join(output_dir, os.path.basename(src_img_path))
    #    shutil.copy(src_img_path, dst_img_path)

    print(f"已将 top 300 张图像复制到目录：{output_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--json_path', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)