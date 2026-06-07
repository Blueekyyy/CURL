from glob import glob
import os
from argparse import ArgumentParser
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import Resize
import torchvision
import pickle as pkl
import sys
#sys.path.append('../../ext/hyperIQA')
#import models
import json


#import pdb; pdb.set_trace()
#transforms = torchvision.transforms.Compose([
#    torchvision.transforms.Resize((512, 288)),
#    torchvision.transforms.RandomCrop(size=224),
#    torchvision.transforms.ToTensor(),
#    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
#                                     std=(0.229, 0.224, 0.225))])

#model_hyper = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7).cuda()
#model_hyper.train(False)
# load our pre-trained model on the koniq-10k dataset
#model_hyper.load_state_dict((torch.load('../../ext/hyperIQA/pretrained/koniq_pretrained.pkl')))


def main(args):
    data_path = args.data_path
    iqa_scores = {}
    basenames = []
    
    #import pdb;pdb.set_trace()
    with open(f"{args.data_path}/aes-input-2.json", "r") as f:
        for line in f:
            data = json.loads(line)
            image_name = data["image"].split("/")[-1]
            MOS_score = data["MOS"]
            iqa_scores[image_name.replace('.png', '')] = MOS_score

    #import pdb;pdb.set_trace()
    for filename in tqdm(glob(f'{data_path}/images/*')):
        basename = os.path.basename(filename).split('.')[0]
        basenames.append(basename)

        img = np.asarray(Image.open(f'{data_path}/images/{basename}.png'))
        mask_hair = np.asarray(Image.open(f'{data_path}/masks/hair/{basename}.png'))
        mask_face = np.asarray(Image.open(f'{data_path}/masks/face/{basename}.png'))
        mask_body = np.asarray(Image.open(f'{data_path}/masks/body/{basename}.png'))

        # Check intersection between face and hair
        if ((mask_hair > 127) * (mask_face > 127)).sum() > (mask_body > 127).sum() * 0.1:
            print(f'Skipping frame {basename}')
            continue

        # Crop
        #h, w = img.shape[:2]
        #i, j = np.nonzero(mask_hair > 0.0)
        #l, r = j.min(), j.max()
        #u, d = i.min(), i.max()
        #sx = r - l
        #sy = d - u
        #px = int(sx * 0.05)
        #py = int(sy * 0.05)
        #l = max(l - px, 0)
        #r = min(r + px, w)
        #u = max(u - py, 0)
        #d = min(d + py, h)
        
        #img = img[u:d, l:r] * (mask_hair[u:d, l:r, None] / 255.)
        #img = Image.fromarray(img.astype('uint8'))

        #pred_scores = []

        #for _ in range(10):
        #    img_tr = transforms(img)  # transform增强。重复10次（因为transform有随机性）
        #    img_tr = torch.tensor(img_tr.cuda()).unsqueeze(0)
        #    with torch.no_grad():
        #        params = model_hyper(img_tr)

            # Building target network
        #    model_target = models.TargetNet(params).cuda()
        #    for param in model_target.parameters():
        #        param.requires_grad = False

            # Quality prediction
        #    pred = model_target(params['target_in_vec'])  # 'paras['target_in_vec']' is the input to target net
        #    pred_scores.append(float(pred.item()))

        #iqa_score_basename = np.mean(pred_scores)
        #if iqa_score_basename > args.iqa_threshold:
        #    iqa_scores[basename] = iqa_score_basename

        #从aes-input.json读取id和['logits']['excellent']分数存到iqa_scores字典



    pkl.dump(iqa_scores, open(f'{data_path}/iqa_scores_hair.pkl', 'wb'))
    # iqa_scores = pkl.load(open(f'{data_path}/iqa_scores_hair.pkl', 'rb'))
    

    #得到
    #import pdb;pdb.set_trace()
    # Split IQA scores into bins according to the frame index 根据帧索引将 IQA 分数分成几箱
    #img_names = sorted(iqa_scores.keys())
    img_names = sorted(basenames)
    frame_idx = np.asarray([int(k) for k in img_names])  #将帧名从字符转为int型id
    print(frame_idx)
    num_bins = args.max_imgs
    
    while True:
        print(f'Trying to split into {num_bins}')
        hist, bins = np.histogram(frame_idx, bins=num_bins)
        print(hist)
        if sum(hist != 0) >= args.max_imgs:  # 不断尝试 bin 数量直到每个bin里面有图，且总bin数够多。
            break
        else:
            num_bins += 1
    print(f'Splitting frames in {num_bins} bins')

    img_names_split = []
    for i in range(num_bins):
        if hist[i]:
            frame_idx_bin = frame_idx[np.logical_and(frame_idx >= bins[i], frame_idx < bins[i + 1])]
            img_names_chunk = []
            for j in frame_idx_bin:
                img_names_chunk.append('%06d.png' % j)
            img_names_split.append(img_names_chunk)
    assert len(img_names_split) >= args.max_imgs
    print(img_names_split)

    #import pdb;pdb.set_trace()
    img_names_filtered = []
    for img_names_chunk in img_names_split:
        iqa_scores_chunk = []
        for img_name in img_names_chunk:
            #iqa_scores_chunk.append(iqa_scores[img_name.replace('.png', '')])
            #=== Test ===
            key = img_name.replace('.png', '')
            if key in iqa_scores:
                iqa_scores_chunk.append(iqa_scores[key])
            else:
                print(f"警告：{key} 不在 iqa_scores 中，跳过")
                # 也可以选择继续或者给个默认值
            
        
        # === Test ===
        if not iqa_scores_chunk :
            continue
        #import pdb;pdb.set_trace()
        img_names_filtered.append(img_names_chunk[np.argmax(np.asarray(iqa_scores_chunk))])  # 在每个bin里挑出最好的一张图
    
    #import pdb;pdb.set_trace()
    pkl.dump(img_names_filtered, open(f'{data_path}/iqa_filtered_names.pkl', 'wb'))  # 把筛选出来的高质量图片名字保存到iqa_filtered_names.pkl

    #=== Test ===
    # === 找出最低 MOS 分数帧，并写入文件 ===
    min_score = float('inf')
    min_name = None

    for name in img_names_filtered:
        key = name.replace('.png', '')
        score = iqa_scores.get(key, None)
        if score is not None and score < min_score:
            min_score = score
            min_name = name

    print(f'\n最低 MOS 分数帧为: {min_name}，分数为: {min_score}')

    # 保存到文件
    with open(f'{data_path}/worst_frame.txt', 'w') as f:
        f.write(f'{min_name}\t{min_score}\n')


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--max_imgs', default=128, type=int)
    # === Test ===
    parser.add_argument('--iqa_threshold', default=10, type=float)
    #parser.add_argument('--iqa_threshold', default=50, type=float)

    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)