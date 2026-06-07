import os
from argparse import ArgumentParser
from openai import OpenAI

def main(args):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8003/v1"


    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    images_dir = os.path.join(args.data_path, "images_2")
    if not os.path.exists(images_dir):
        print(f"目录不存在: {images_dir}")
        return
    
    max_complexity = 0  # 保存最高复杂度
    
    # 单图像输入推理
    #image_url = "file:///mnt/data/ljy/Li/GaussianHaircut/Input/20250717_i_wig_normal_light/images/000001.png"
    for filename in sorted(os.listdir(images_dir)):
        if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            continue
        image_path = os.path.join(images_dir, filename)
        image_url = f"file://{image_path}"


        prompt_text = (
            "您是一位数据评估员，专门负责根据图像判断头发的复杂度（从直发到微卷、大卷、爆炸卷、麻花辫，从短发再到长发，复杂度逐步递增）并评分。"
            "您将获得一张图片。您的任务是从一个方面，以 5 分制评估头发的复杂程度：头发复杂度等级参考（复杂度递增顺序）\n"
            "1 直发：发丝笔直，下垂自然，90%没有弯曲、卷曲或波浪，尾部少量卷曲，层次感较少，整体平滑\n"
            "2 自然卷：天生卷曲，不规则弯曲，每根发丝的卷度可能不同，有时呈波浪或小弯钩，卷曲方向多样，局部易打结\n"
            "3 大波浪卷：卷度宽松，呈现出大弧度波浪，卷曲方向整体一致，曲线流畅，发丝通常按层次排列，规律感较强\n"
            "4 小卷：卷度比大波浪卷更小，卷曲更紧密，发丝呈现螺旋状或弹簧状，卷圈直径较细，卷曲方向较统一，但密度比大卷高\n\n"
            "5 爆炸卷：卷度极小，呈现 高密度螺旋状或Z字形弯曲，发丝蓬松向外扩张，整体呈现“球形”或“云状”，每根发丝卷曲方向复杂，缺乏统一性\n"
            "6 麻花辫：由三股或更多发丝相互交织编织而成，呈现出规律的交叉结构，线条感清晰，可以是单条辫子，也可以是多股小辫子组合\n"
            "7 鱼骨辫：由两股头发反复交叉小撮发丝编织而成，交织方式像鱼骨或麦穗，辫子线条比传统麻花辫更细腻、更紧密\n"
            "8 混合发型: 由麻花辫、卷发、直发、马尾辫组合成的发型，形状遍布在头上的各个地方\n"
            "示例响应格式：\n简要理由：根据上述标准对分数进行简短解释，不超过 50 个字。\n"
            "头发复杂度等级参考：从1到8的数字。\n"
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
        #print("Chat completion output:", chat_response.choices[0].message.content)

        # 提取模型返回的复杂度等级
        response_text = chat_response.choices[0].message.content
        print(f"{filename} -> {response_text}")

        # 尝试从文本中提取数字（1~5）
        complexity_level = 0
        for line in response_text.splitlines():
            if "头发复杂度等级参考" in line:
                try:
                    complexity_level = int(line.split("：")[-1].strip())
                except:
                    pass
        if complexity_level > max_complexity:
            max_complexity = complexity_level
            print(f"max_complexity: {max_complexity}")

    # 保存最高复杂度到 complex.txt
    with open(os.path.join(args.data_path, "complex.txt"), "w") as f:
        f.write(str(max_complexity))
    print(f"最高头发复杂度等级参考: {max_complexity}，已保存到 complex.txt")

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)

"""
prompt_text = (
            "您是一位数据评估员，专门负责根据图像判断头发的复杂度（从直发到微卷、大卷、爆炸卷、麻花辫，从短发再到长发，复杂度逐步递增）并评分。"
            "您将获得一张图片。您的任务是从一个方面，以 5 分制评估头发的复杂程度：头发复杂度等级参考（复杂度递增顺序）\n"
            "1 直发：最简单，发丝单向排列，层次少。\n"
            "2 微卷：轻微波浪形状，局部卷曲。\n"
            "3 大卷：明显大波浪，发丝层次较丰富。\n"
            "4 爆炸卷：高度卷曲，密度大，方向复杂。\n"
            "5 麻花辫：多股辫子交织，结构清晰，复杂度高。\n\n"
            "示例响应格式：\n简要理由：根据上述标准对分数进行简短解释，不超过 20 个字。\n"
            "头发复杂度等级参考：从1到5的数字。\n"
        )
"""