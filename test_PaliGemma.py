import torch
import base64
import os
from util import load_annotations, get_easy_medium_hard_items, local_image_to_data_url

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")


def test_pali_gemma(prompt: str, image_path: str):
    """
    Test the PaliGemma model with a sample image and prompt.
    """

    from transformers import pipeline
    pipeline = pipeline(
            task="image-text-to-text",
            model="google/paligemma2-3b-mix-448",
            device=0,
            torch_dtype=torch.bfloat16
        )

    # Load the image and convert it to base64
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    result = pipeline(
        images=[f"data:image/jpeg;base64,{b64_image}"],
        text=prompt,
    )
    print(result[0]['generated_text'])
    return result[0]['generated_text']



def test_Qwen3B(prompt: str, image_path: str):
    with open(image_path, "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{b64_image}",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # print(output_text)
    return output_text[0].strip()




if __name__ == "__main__":
    data = load_annotations("data_ref/instruction_gpt4_all.json")
    easy_ids, medium_ids, hard_ids = get_easy_medium_hard_items(data)
    data_grps = {'easy': easy_ids, 'medium': medium_ids, 'hard': hard_ids}

    p1 = f"<image> What is in this image?"
    p2 = f"<image> You are a helpful assistant that will help me in a scientific study. Please work reliably.\
            Please give me the list of objects with their corresponding numbers as labeled in the image. \
            Output in dictionary format like '1': 'box', '2': 'bottle'..  Only output the dictionary of objects."
    p3 = f"<image> Output the list of objects with their corresponding numbers as labeled in the image. \
            Output in dictionary format like '1': 'box', '2': 'bottle'.." #Only output the dictionary of objects."
    p4 = f"<image> list all objects in the image: " 
    prompt = p4  # re-run Sep 16, for camera-ready.


    for item in easy_ids:
        image_path = f"data_ref/Images_NumPrompts/{item}"
        if not os.path.exists(image_path):
            continue
        print(f"Testing item: {item}")
        #result =test_pali_gemma(prompt, image_path)
        
        result = test_Qwen3B(prompt, image_path)
        objects = data[item]['objects']
        print(f"Objects in the image: {objects}")
        print(f"Result: {result}")
        print("-" * 50)