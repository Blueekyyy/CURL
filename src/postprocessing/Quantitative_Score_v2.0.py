import os
from argparse import ArgumentParser
from openai import OpenAI

def main(args):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    
    images_dir = os.path.join(args.data_path, args.directory_name)
    if not os.path.exists(images_dir):
        print(f"目录不存在: {images_dir}")
        return

    # ------------------------------
    # 1. 读取第一张原始图像作为 Reference Image，读取第二张剪影图像作为 Mask Image
    # ------------------------------
    #因为是按字典序排序，目录中的图像总数要小于10才准确
    #目录图像命名格式为：序号-算法-视图
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(image_files) == 0:
        print("没有找到图像")
        return

    ref_name = image_files[0]
    ref_path = os.path.join(images_dir, ref_name)
    ref_url = f"file://{ref_path}"

    mask_name = image_files[1]
    mask_path = os.path.join(images_dir, mask_name)
    mask_url = f"file://{mask_path}"

    print(f"选定参考图像 Reference Image: {ref_name}")
    print(f"选定剪影图像 Mask Image: {mask_name}")

    # ------------------------------
    # 2. 输出文件
    # ------------------------------
    output_path = os.path.join(args.data_path, args.txt_name)
    with open(output_path, "w") as f:

        # ------------------------------
        # 遍历所有重建图片
        # ------------------------------
        for filename in image_files[2:]:

            cur_path = os.path.join(images_dir, filename)
            cur_url = f"file://{cur_path}"

            # ---------------------------------------------
            # Prompt：让模型比较 “ref原始图像” + “mask剪影图像” vs “当前重建图像”
            # ---------------------------------------------
            prompt_text = (
                "你是一个头发重建质量评估系统，需要比较：\n"
                "1. 参考原图（第一张）\n"
                "2. 剪影图（第二张）\n"
                "3. 当前重建图（第三张）\n\n"
                "请基于以下标准对重建图从 0-100 打分：\n"
                "① 重建图的头发轮廓是否符合剪影图\n"
                "② 重建图的头发纹理方向与参考原图是否一致\n"
                "只输出一个分数 + 几句理由。"
                "几句理由要包括：描述图像的特征，mask大约什么形状，输出头发的图片或给个头发位置的坐标"
            )

            # ---------------------------------------------
            # 关键修改：一次传三张图 → Qwen2.5-VL会对比
            # ---------------------------------------------
            chat_response = client.chat.completions.create(
                model="qwen2.5-vl-7b",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},

                            # 第一张图：参考原图
                            {"type": "text", "text": "【参考原图】"},
                            {"type": "image_url", "image_url": {"url": ref_url}},

                            # 第二张图：剪影图
                            {"type": "text", "text": "【剪影图】"},
                            {"type": "image_url", "image_url": {"url": mask_url}},

                            # 第三张图：当前重建图
                            {"type": "text", "text": "【待评估重建图】"},
                            {"type": "image_url", "image_url": {"url": cur_url}},
                        ],
                    }
                ],
            )

            # 提取模型返回内容
            response_text = chat_response.choices[0].message.content
            line = f"{filename} -> {response_text}"
            print(line)
            f.write(line + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--directory_name', default='', type=str)
    parser.add_argument('--txt_name', default='', type=str)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()
    main(args)
