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
    
    min_quality = 100  # 保存最高复杂度
    
    # 单图像输入推理
    #image_url = "file:///mnt/data/ljy/Li/GaussianHaircut/Input/20250717_i_wig_normal_light/images/000001.png"
    for filename in sorted(os.listdir(images_dir)):
        if not (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")):
            continue
        image_path = os.path.join(images_dir, filename)
        image_url = f"file://{image_path}"


        prompt_text = (
            "您是一位数据评估员，专门负责根据图像判断图像的光照质量（从昏暗、过曝到清晰、自然、精致，从不均匀、高对比度再到理想，质量逐步递增）并评分。"
            "您将获得一张图片。您的任务是从一个方面，以 18 分制评估图像的光照质量：光照质量等级参考（质量递增顺序）\n"
            "1 昏暗的：图像整体呈现低亮度，细节不清晰，通常很难看清物体的形状或纹理。常见于光源不足的环境，或者光源距离物体较远，导致画面整体缺乏亮度。\n"
            "2 不足的：虽然有一定的光源，但亮度依然不足，画面无法完全照亮所有细节。局部区域可能明显较暗，尤其是在光源不够强烈或被遮挡的情况下。\n"
            "3 不均匀的：光照分布不均，某些区域过亮，另一些区域则过暗。常见于光源方向不当或多光源照射下。画面可能看起来有明显的阴影区域或反光区域。\n"
            "4 阴暗的：图像整体看起来灰蒙蒙的，没有鲜明的亮度对比。尽管光源在场，但未能有效提升图像的明亮度和活力。通常表现为缺乏色彩饱和度和活力。\n"
            "5 模糊的：由于光照过于分散或不够集中，图像出现了一定程度的模糊，尤其在焦点区域。光照不足以清晰定义物体的边缘和细节。\n"
            "6 刺眼的：光照强烈且集中，导致某些区域过亮甚至产生强烈反射或炫光。常见于直射阳光或强光源照射下，可能会导致细节丢失。\n"
            "7 不自然的：光照看起来过于人工，可能是人造光源如霓虹灯、荧光灯等。光的色温和分布不符合自然环境，使得物体呈现出不自然的色彩和亮度。\n"
            "8 过曝的: 图像中的亮部区域过于亮，甚至完全丧失了细节。亮部细节被溢出或烧焦，通常是因为光源过强或曝光时间过长。\n"
            "9 高对比度的: 图像的亮度差异非常大，暗部和亮部之间的对比明显。虽然看起来有较好的视觉冲击力，但可能会丢失一些中间细节。\n"
            "10 柔和的: 光照分布均匀，亮度适中。常见于日光或阴天等柔和的光源下，物体的阴影过渡平滑，色温温和自然。\n"
            "11 平衡的: 图像的光照非常均匀，亮度和对比度都经过良好的调节，既不会过亮也不会过暗。物体的细节得以清晰呈现，且没有明显的高光或阴影问题。\n"
            "12 清晰的: 光照明亮且清晰，细节呈现得非常完美。每个物体的轮廓和结构都可以清晰辨识，通常这种效果适用于高质量的光源和正确的曝光。\n"
            "13 明亮的: 图像的亮度很高，整体看起来明亮且充满活力。通常是在光源强烈或环境光照较好的情况下产生。\n"
            "14 自然的: 光照呈现自然的效果，类似于自然日光或环境光，颜色温和且不会过于突出某个区域。物体的颜色和亮度符合我们的自然感知。\n"
            "15 温暖的: 光照带有温暖的色调，通常是黄色、橙色或红色的光源。常见于日落、蜡烛或室内灯光等环境中，营造温馨的氛围。\n"
            "16 理想的: 光照完美，亮度适中，阴影和高光平衡得当。没有过度曝光或欠曝光的情况，能够完美展示物体的细节和色彩。\n"
            "17 精致的: 光照非常精致，细节和纹理清晰，阴影过渡自然。每个细节都得到细心处理，呈现出高艺术感。\n"
            "18 完美的: 光照在所有方面都近乎完美，既有适当的亮度，又有良好的阴影效果。物体的质感、颜色和细节都得到了最佳展示。\n"
                "示例响应格式：\n简要理由：根据上述标准对分数进行简短解释，不超过 50 个字。\n"
                "光照质量等级参考：从1到18的数字。\n"
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
        quality_level = 0
        for line in response_text.splitlines():
            if "光照质量等级参考" in line:
                try:
                    quality_level = int(line.split("：")[-1].strip())
                except:
                    pass
        if quality_level < min_quality:
            min_quality = quality_level
            name = filename
            print(f"min_light_quality: {min_quality}, name: {name}")

    # 保存最高复杂度到 complex.txt
    with open(os.path.join(args.data_path, "worst_frame_Qwen2.5-VL.txt"), "w") as f:
        #f.write(str(min_quality))
        f.write(f'{name}\t{min_quality}\n')
    print(f"最低光照质量等级参考: {min_quality}，已保存到 worst_frame_Qwen2.5-VL.txt")

if __name__ == "__main__":
    parser = ArgumentParser(conflict_handler='resolve')

    parser.add_argument('--data_path', default='', type=str)
    
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    main(args)