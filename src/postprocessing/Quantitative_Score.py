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
    
    #20张以内视图
    #正视图、侧视图、后视图各搞一个目录
    images_dir = os.path.join(args.data_path, "Quantitative")
    if not os.path.exists(images_dir):
        print(f"目录不存在: {images_dir}")
        return
    
    # 单图像输入推理
    output_path = os.path.join(args.data_path, "Quantitative_Score.txt")
    with open(output_path, "w") as f:
        for filename in sorted(os.listdir(images_dir)):
            if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
                continue
            image_path = os.path.join(images_dir, filename)
            image_url = f"file://{image_path}"


            prompt_text = (
                "略，后续再补"
            )


            chat_response = client.chat.completions.create(
                model="qwen2.5-vl-7b",
                messages=[{
                    "role": "user",
                    "content": [
                        # NOTE: The prompt formatting with the image token `<image>` is not needed
                        # since the prompt will be processed automatically by the API server.
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }],
            )

            # 提取模型返回的定量等级
            response_text = chat_response.choices[0].message.content
            line = f"{filename} -> {response_text}"
            print(line)
            f.write(line + "\n")   # 逐行写入同一个文件

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)