"""
Using multimodal LLMs for object recognition in robot manipulation task.  (written by: X.Sui) 
"""
import os
import json
import pprint
import ast
from collections import defaultdict
import base64
import numpy as np
from mimetypes import guess_type 
import PIL
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from google import genai
from google.genai import types
import openai
from openai import AzureOpenAI 
from together import Together
import ollama
import Janus  #deepseek
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
import dashscope  # aliyun
dashscope.api_key = "add your key here"

import util
from util import load_annotations, get_easy_medium_hard_items, local_image_to_data_url, draw_candidate_boxes, get_unique_items


def GroundingDINO(image_url, text_labels):
    image_pil = Image.open(image_url).convert("RGB") 
    model_id = "IDEA-Research/grounding-dino-tiny"
    #model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image_pil, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image_pil.size[::-1]]
    )
    result = results[0]
    return result

# initialize model
def initialize_ds_chat(FM_model="Janus-Pro-7B"):
    print(f"Initializing {FM_model}")
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.float16).cuda().eval()
    return vl_chat_processor, tokenizer, vl_gpt
#vl_chat_processor, tokenizer, vl_gpt = initialize_ds_chat("Janus-Pro-7B")

# Query MLLMs
def query_deepseek_with_image(prompt, image_url):
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{prompt}",
            "images": [image_url],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device, dtype=torch.float16)

    # # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    #print(f"{prepare_inputs['sft_format'][0]}", answer)
    return answer

def query_aliyun_with_image(prompt, image_url):
    messages = [{
        'role': 'user',
        'content': [
            {'image': image_url},
            {'text': prompt},
        ]
    }]
    response = dashscope.MultiModalConversation.call(model='qwen2.5-vl-72b-instruct', messages=messages)
    answer = response.output.choices[0].message.content[0]['text']
    return answer

def query_mFM_with_image(prompt, image_url, FM_model):
    gptformat_list = ["gpt4vision", "gpt-4o-mini-20240718",  "gpt4o0513",
                    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
                    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "Qwen/Qwen2-VL-72B-Instruct"] 

    if FM_model in gptformat_list:
        # Query Azure OpenAI GPT-4o with extracted image data.
        response = client.chat.completions.create(
            model= FM_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", 
                    "text": prompt,
                    },
                    {"type": "image_url",
                    "image_url": {
                        "url": image_url,
                        },
                    }
                ],
            }],
            max_tokens=3000,
        )
        print(response.choices[0])
        return response.choices[0].message.content
    elif FM_model in ["phi-3.5"]:
        from azure.ai.inference.models import TextContentItem, ImageContentItem, ImageUrl
        response = client.complete(
            messages=[
                SystemMessage(content="You are a helpful assistant that can generate responses based on images."),
                UserMessage(content=[
                    TextContentItem(text = prompt),
                    ImageContentItem(image_url = ImageUrl(url=image_url))
                ]),
            ],
            temperature=0,
            top_p=1,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    elif FM_model in ["llama3.2-vision", "llama3.2-vision:90b"]:
        response = ollama.chat(
            model=FM_model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_url]
            }]
        )
        return response.message.content
    elif FM_model in ["Janus-Pro-7B"]:
        response = query_deepseek_with_image(prompt, image_url)
        return response
    elif FM_model in ["Qwen2.5-VL-72B"]:
        response = query_aliyun_with_image(prompt, image_url)
        return response
    elif FM_model in ["Gemini2.5Pro", "Gemini2.0Flash"]:
        image = PIL.Image.open(image_url)
        client = genai.Client(api_key = "add your key") 
        response = client.models.generate_content(
            #model="gemini-2.5-pro-exp-03-25",
            model = "gemini-2.0-flash",
            contents=[prompt, image])
        return response.text

def query_mFM_with_text(client, prompt, FM_model):
    # Query Azure OpenAI or Together Llama with text data.
    response = client.chat.completions.create(
        model= FM_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", 
                "text": prompt,
                },
            ],
        }],
        max_tokens=3000,
    )
    print(response.choices[0])
    return response.choices[0].message.content

def query_gpt45_with_image(image_path, prompt):
    import requests
    # image is base64 encoded
    API_KEY = 'ADD YOUR API KEY HERE'
    def encode_image(image_url):
        with open(image_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    encoded_image = encode_image(image_url=image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "gpt-4.5-preview-2025-02-27",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        "max_tokens": 500
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        answer = result['choices'][0]['message']['content']
        return answer
    else:
        raise Exception(f"Request failed: {response.status_code} {response.text}")
    return answer


# Objects Grounding 
def identify_objects(image_url, FM_model):
    print(f"Object identification")
    if FM_model in ["GroundingDINO"]:
        # Use bbox and number prompts to evaluate detection results.
        prompt = f"The image is the result of an object detection model. Each detected object is enclosed within a bounding box, and \
        bounding box usually contains a number. Please list the detected object labels in the order of the numbers shown inside the boxes. \
        Format the answer as: 1: object_name, 2: object_name, etc. If a numbered object is not detected or labeled, write false for that number. \
        Output in dictionary format.  Example: '1': marker, '2': 'bottle',  '3': 'false', ...Only output the dictionary of objects."
        answer = query_mFM_with_image(prompt, image_url, FM_model='gpt4o0513')
    else:
        # only use number prompts for generalist models.
        prompt = f"You are a helpful assistant that will help me in a scientific study. Please work reliably. \
            Please give me the list of objects wih their corresponding numbers as labeled in the image. \
            Output in dictionary format like '1': 'box', '2': 'bottle'..  \
            Only output the dictionary of objects."

        if FM_model == 'gpt4.5':
            answer = query_gpt45_with_image(image_url, prompt)
        else:
            answer = query_mFM_with_image(prompt, image_url, FM_model)


    try:
        answer_dict = ast.literal_eval(answer)
        return answer_dict  
    except:
        print("Error: object identification answer is not in dictionary format.")
        return answer  

def gpt4_verify_objects(response, grd_truth_dict, FM_model="gpt4vision"):
    print(f"Object Verification")

    client = AzureOpenAI(
            api_key = "ADD Your Key Here",  
            api_version = "2024-02-15-preview",
            azure_endpoint="https://openai-nrp-robot-sui.openai.azure.com/",
            )
    # Verify the response from FMs with the ground truth objects
    prompt = f"Please compare the lists of objects with their corresponding numbers. Ground truth list is provided below. {grd_truth_dict}. \
        The estimated list is {response}. \
        Compare each item, if the item is semantically correct, which indicates the same object category, use 1 as flag; otherwise 0. \
        e.g.: 'marker' and 'boardmarker' are the same category, so the flag is 1.  'cup' and 'mug' are semantically the same; 'can' and 'tin' are semantically the same.\
        Only output the list of the flags in the same order as the objects; e.g., [1, 0, 1, 1, 0, 1, 1, 1, 1, 1]. \
        Do not explain the reasoning, only output the list of flags." 
    
    answer = query_mFM_with_text(client, prompt, FM_model)
    try:
        result_list = ast.literal_eval(answer)
    except:
        print("Error: Object Verification answer is not in list format.")
    return result_list


if __name__ == "__main__":
    # Load Data
    data = load_annotations("data_ref/instruction_gpt4_all.json")
    easy_ids, medium_ids, hard_ids = get_easy_medium_hard_items(data)
    data_grps = {'easy': easy_ids, 'medium': medium_ids, 'hard': hard_ids}

    # Set Foundation Models
    # Specialists
    Specialists = ["GroundingDINO"] 

    # Generalist
    Azure_FMs = ["gpt4vision", "gpt-4o-mini-20240718",  "gpt4o0513", "phi-3.5"]  
    Together_FMs = ["meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
                    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "Qwen/Qwen2-VL-72B-Instruct"]
    ollama_FMs = ["llama3.2-vision", "llama3.2-vision:90b"]
    deepseek_FMs = ["Janus-Pro-7B"]
    Aliyun_FMs = ["Qwen2.5-VL-72B"]
    Gemini_FMs = ["Gemini2.5Pro", "Gemini2.0Flash"]
    New_FMs = ["gpt4.5"]
    FM_oformat = Azure_FMs[0] 

    # Testing
    FM_testing = Gemini_FMs[1]
    FM_savefile = Gemini_FMs[1]
    print(f"Testing Foundation Model: {FM_testing} \n")

    # Initialize API via dsAzure or Together
    if FM_testing in ["gpt4vision", "gpt-4o-mini-20240718"]:
        client = AzureOpenAI(
            api_key = "ADD Your Key Here",  
            api_version = "2024-02-15-preview",
            azure_endpoint="https://openai-nrp-robot-sui.openai.azure.com/",
            )
    elif FM_testing in ["gpt4o0513"]:
        client = AzureOpenAI(
            api_key= "ADD Your Key Here",
            api_version = "2024-08-01-preview",
            azure_endpoint="https://openai-kgllm-instruct.openai.azure.com/"
            )
    elif FM_testing in ["phi-3.5"]:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        keys = { 'llama': ' ADD Your Key Here',
                'phi35': 'ADD Your Key Here',
                'gpt4o': 'ADD Your Key Here' }

        model_names = { 'llama': 'Llama-3.2-90B-Vision-Instruct',
                        'phi35': 'Phi-3.5-vision-instruct',
                        'gpt4o': 'gpt-4o' }

        endpoints = { 'llama': 'ADD Your MODEL ENDPOINT Here',
                    'phi35': 'ADD Your MODEL ENDPOINT Here',
                    'gpt4o': 'ADD Your MODEL ENDPOINT Here' }
        api_versions = { 'llama': '2024-05-01-preview',
                        'phi35': '2024-05-01-preview',
                        'gpt4o': '2024-08-01-preview' }

        model_id = 'phi35'
        client = ChatCompletionsClient(
            endpoint=endpoints[model_id],
            credential=AzureKeyCredential(keys[model_id]),
            model=model_names[model_id],
            api_version=api_versions[model_id],
            max_tokens=1000
        )
    elif FM_testing in Together_FMs:
        # Together client
        os.environ['TOGETHER_API_KEY'] = "ADD Your KEY Here"
        client = Together()
    elif FM_testing in ollama_FMs:
        print("Testing Ollama Foundation Models")
    elif FM_testing in deepseek_FMs:
        print("Testing Deepseek Janus Pro 7B Foundation Model")
    elif FM_testing in Aliyun_FMs:
        print("Testing Aliyun Qwen2.5-VL Foundation Models")
    elif FM_testing in Gemini_FMs:
        print("Testing Gemini Foundation Models")

    # Object Identification Test
    results = defaultdict(dict)

    for difficulty_level in ['easy', 'medium', 'hard']:
    #for difficulty_level in ['easy']:
        print(f"-------------{difficulty_level}: {len(data_grps[difficulty_level])} images ------\n")
        # initialize results
        macro_acc = {'obj': 0.0}
        macro_counter = {'obj':0}
        missing_ids = {'object':[]}
        
        grounding = defaultdict(lambda: defaultdict(dict))

        # iterate subsets
        for img_id in data_grps[difficulty_level]:

            if FM_testing in ["gpt4.5","llama3.2-vision", "llama3.2-vision:90b", "Janus-Pro-7B", "GroundingDINO", "Gemini2.5Pro","Gemini2.0Flash"]:#,"Qwen2.5-VL-72B"]:
                imageNum_url = f"data_ref/Images_NumPrompts/{img_id}"
            else:
                imageNum_url = local_image_to_data_url(f"data_ref/Images_NumPrompts/{img_id}") 
            print(f"Testing {img_id} \n")
            
            # Object Identifications
            ann_obj = data[img_id]['objects'] 
            if FM_testing in ["GroundingDINO"]:
                # detection results
                imageNumDet_url = f"output_dinobox/det_{img_id}"
                if not os.path.exists(imageNumDet_url):
                    ann_obj_list = get_unique_items(ann_obj)
                    result = GroundingDINO(imageNum_url, ann_obj_list) 
                    draw_candidate_boxes(imageNum_url, result, save_file=f"output_dinobox/det_{img_id}")

                # use gpt-4o to verify the detected objects.
                client = AzureOpenAI(
                    api_key= "add your key here",
                    api_version = "2024-08-01-preview",
                    azure_endpoint="add ur endpoint here"  
                    ) 
                detimg_url = local_image_to_data_url(imageNumDet_url)
                identified_objects = identify_objects(detimg_url, FM_testing)
                verification_flags = gpt4_verify_objects(identified_objects, ann_obj, "gpt4vision")
                macro_counter['obj'] += 1
            else:
                try:
                    identified_objects = identify_objects(imageNum_url, FM_testing)
                    verification_flags = gpt4_verify_objects(identified_objects, ann_obj, "gpt4vision")
                    macro_counter['obj'] += 1
                except Exception as e:
                    print(f"Error: Missing data for image {img_id}. Exception raised {e}.")
                    missing_ids['object'].append(img_id)
                    continue

            grounding[img_id]['object']['grd_truth'] = ann_obj
            grounding[img_id]['object']['identified'] = identified_objects
            grounding[img_id]['object']['flags'] = verification_flags
            grounding[img_id]['object']['acc'] = sum(verification_flags) / len(verification_flags)  
            macro_acc['obj'] += grounding[img_id]['object']['acc']    


        # MacroAcc for each difficulty level
        results[difficulty_level] = grounding
        results['missing_ids'] = missing_ids
        macro_acc = {key: macro_acc[key] / macro_counter[key] for key in macro_acc}
        results[difficulty_level]['macro_acc'] = macro_acc
        # pprint.pprint(grounding) 
        # pprint.pprint(macro_acc)
        print(f"Macro Accuracy: {results[difficulty_level]['macro_acc']}\n")
        
        # Save results to a JSON file
        with open(f"{FM_savefile}_obj_results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)

    # final results 
    table_results = defaultdict(dict)
    for difficulty_level in ['easy', 'medium', 'hard']:
        table_results[difficulty_level] = results[difficulty_level]['macro_acc']

    with open(f"{FM_savefile}_obj_table_results.json", "w") as json_file:
        json.dump(table_results, json_file, indent=4)
    print(f"Grounding data has been saved to json files.")
