import os
from PIL import Image
from scorer import QAlignScorer
import json
from argparse import ArgumentParser

#root_dir = "/mnt/data/ljy/Li/Test/Q_Align_test_Images"

scorer = QAlignScorer()

# 简单的评分到评价映射函数
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

def make_json(image_paths, image_scores, data_path, mode):
    image_score_list = []
    for path, score in zip(image_paths, image_scores):
        image_score_list.append((path, score))

    # 构建 JSON 数据
    output_data = []

    for image_path, score in image_score_list:
        item = {
            "id": f"{image_path}->0.000000",
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "How would you rate the quality of this image?\n<|image|>"
                },
                {
                    "from": "gpt",
                    #"value": score_to_text(score)
                    "value": f"The quality of the image is to be evaluated."
                }
            ],
            "gt_score": 0.000000
        }
        output_data.append(item)

    #with open("test_1.json", "w") as f:
    #import pdb;pdb.set_trace()
    output_path = os.path.join(data_path, f"{mode}.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)


def get_image_paths(folder, exts={'.png', '.jpg', '.jpeg'}):
    image_paths = []
    for root, _, files in os.walk(folder):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def get_image_score_list(subfolder_path):

    image_path = get_image_paths(subfolder_path)
    #image_path = image_path[:250]

    # IQA
    #img_names = os.listdir(f'/mnt/data/ljy/Li/GaussianHaircut/input_i_wig_normal_light_0/images')
    image_score_list = []
    batch_size = 250

    for i in range(0, len(image_path), batch_size):
        print(f"now i = {i}")
        img_list = []
        batch_paths = image_path[i:i+batch_size]
        for img_name in batch_paths:
            img_list.append(Image.open(img_name))

        for j in range(0, len(img_list), 50):   # 修改
            print(f"j = {j} to {j + 50}")
            batch = img_list[j:j+50]
            #print(f"makeing score of {i + j} to {i + j + 50}")
            scores = scorer(batch).tolist()
            image_score_list += scores
        
    return image_path ,image_score_list

def main(args):
    #import pdb;pdb.set_trace()
    image_paths = []
    image_scores = []
    root_dir = args.data_path # 修改
    data_path = args.data_path
    mode = args.mode

    #import pdb; pdb.set_trace()
    #for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, f"{args.subfolder}")
    image_path, image_score = get_image_score_list(subfolder_path)
    image_paths += image_path
    image_scores += image_score
    
    #import pdb;pdb.set_trace()
    make_json(image_paths, image_scores, data_path, mode)

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--mode', default='', type=str)
    parser.add_argument('--subfolder', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)