"""
Using multimodal LLMs for goal specification in robot manipulation task.  (written by: X.Sui) 
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
import time 

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

from google import genai
from google.genai import types
import openai
from openai import AzureOpenAI 
from together import Together
import ollama
import Janus # deepseek
from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
from Janus.janus.utils.io import load_pil_images
import dashscope # aliyun
dashscope.api_key = "add your key here"

import util
from util import load_annotations, get_easy_medium_hard_items, local_image_to_data_url
from util import draw_candidate_boxes,  draw_overlay_caption, string_of_empty_dict


def GroundingDINO(image_url, text_labels):
    image_pil = Image.open(image_url).convert("RGB") 
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    inputs = processor(images=image_pil, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
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

# Query Text Format
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

def gpt4_verify_answer_format(LLM_answer, FM_model="gpt4vision"):
    client = AzureOpenAI(
            api_key = "add your key here",  
            api_version = "2024-02-15-preview",
            azure_endpoint="https://openai-nrp-robot-sui.openai.azure.com/",
            )
    # Verify the response from FMs with the ground truth objects
    prompt = f"You are a helpful assistant to evaluate the answer of a Object Referring Task. \
        The answer is： {LLM_answer}. \
        Only Output the list of the object labels in the same order as in the answer , like [4, 5, 6].\
        if the answer includes the explanations, please ignore. only output the list of integers. "
    
    answer = query_mFM_with_text(client, prompt, FM_model)
    try:
        result_list = ast.literal_eval(answer)
    except:
        print("Error: Object Verification answer is not in list format.")
    return result_list

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

def query_mFM_with_image(prompt, image_url, FM_model, verifymode=False):
    if verifymode:
        FM_model = "gpt4o0513"
        client = AzureOpenAI(
                api_key= "add your key here",
                api_version = "2024-08-01-preview",
                azure_endpoint="https://openai-kgllm-instruct.openai.azure.com/"
                )
        image_url = local_image_to_data_url(image_url)
        
    clients_with_sameUI = ["gpt4vision", "gpt-4o-mini-20240718",  "gpt4o0513",
                    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
                    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "Qwen/Qwen2-VL-72B-Instruct"]
    
    if FM_model in clients_with_sameUI:
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
                'images': [image_url] #this is image_path string, not base64.
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
        client = genai.Client(api_key = "ADD Your key Here")  
        response = client.models.generate_content(
            model="gemini-2.5-pro-exp-03-25",
            #model = "gemini-2.0-flash",
            contents=[prompt, image])

        return response.text

def query_gpt45_with_image(image_path, prompt):
    import requests
    # image is base64 encoded
    API_KEY = 'ADD Your key Here'
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


# Specialist FMs: Instruction Parsing 
def llm_parse_instruction(instruction, FM_model="gpt4vision"):
    client = AzureOpenAI(
            api_key = "ADD Your key Here",  
            api_version = "2024-02-15-preview",
            azure_endpoint="https://openai-nrp-robot-sui.openai.azure.com/", #Add your endpoint here
            )
    prompt = "You are a robot designed to understand human instructions and identify object references for visual detection.\
            Given an instruction, extract the relevant object names. If there are multiple objects, separate them using commas.  \
            Return the result as a dictionary with the following format:  \
                - \"target\": the main object being referred to or requested.  \
                - \"neighbors\": any reference objects used to help locate the target.  \
                Example:  \
                Instruction: \"Give me the red ball on the left of the white cup.\"  \
                {\"target\": \"red ball\", \"neighbors\": \"white cup\"} \
                Instruction: \"Please pass me the tool that's used for tightening or loosening screws.\" \
                {\"target\"\"the tool that's used for tightening or loosening screws.\, \"neighbors\": \"\"} \
                Usually, there is only one target object, but there can be multiple or none neighbor objects. \
                only output the dictionary, strictly follow the format. Dictionary Format: {\"target\": \"\", \"neighbors\": \"\"} "
    

    prompt += f"Human instruction: {instruction}"
    answer = query_mFM_with_text(client, prompt, FM_model)
    if False:
        print('\n')
        print(instruction)
        print(answer)
        print('\n')
    
    try:
        parsed_dict = ast.literal_eval(answer)
        return parsed_dict
    except:
        print(f"Error: {instruction}: LLM parsing result is not a dictionary!  {answer}")
        return answer 

def llm_select_target(instruction, det_result, FM_model="gpt4vision"):
    client = AzureOpenAI(
            api_key = "add your key here",  
            api_version = "2024-02-15-preview",
            azure_endpoint="https://openai-nrp-robot-sui.openai.azure.com/",
            )
    prompt = "You are a robot designed to understand human instructions and select target object from visual detection results.\
        The detection result is a dictionary that contains the detected objects, and give information on label and boxes. \
        Given the instruction and the detected objects, please select the target object from the detected objects. \
        If the target exisit, select the target, and output the filtered dictionary. \
        If the target does not exist, output an empty dictionary. Output dictionary only! Strictlly Follow the Format! "
    
    prompt += f"Human instruction: {instruction}; Detected Objects: {det_result}"
    answer = query_mFM_with_text(client, prompt, FM_model)

    # answer is a dict in string format, use util funcs to parse it.
    if string_of_empty_dict(answer) or answer == "":
        answer_dict ={}
    else:
        answer_dict = util.parse_tensor_string(answer)
        if not isinstance(answer_dict, dict):
            print(f"Error: LLM select target result is not a dictionary!  {answer_dict}")
    return answer_dict

def mllm_identify_obj_num(image_path, FM_model="gpt4o0513"):
    # Use bbox and number prompts to evaluate detection results.
    prompt = f"This is an object detection result. Each bounding box contains a number corresponding to an object in the image. \
    Please output the detected label for the object with the number. For example 5. \
    if there is no bounding box in the figure, please output 0; \
    if there are more than one bounding boxes, please output 99. Only output number within the bounding box, skip any text, stricktly follow the format. "

    answer = query_mFM_with_image(prompt, image_url=image_path, FM_model="gpt4o0513", verifymode=True)
    try:
        answer = ast.literal_eval(answer.strip())
        if isinstance(answer, int):
            answer = [answer]
        elif isinstance(answer, float):
            answer = [int(answer)]
    except:
        answer = gpt4_verify_answer_format(answer)
    return answer  


# Generalist FMs: Instruction Grounding 
def instruction_grounding(image_url, FM_model, user_instruction=""):
    prompt = f"You are a robot perceiving the visual scene, your job is to find the target requestion by human. \
        Given the instruction, please identify the object in the image, and provide the corresponding number. \
        Only output the number of the object. if there are multiple user instructions, the output format is a list of numbers.\n"
    
    if user_instruction != "":
        prompt += f"Human instruction: {user_instruction}"
    if FM_model == 'gpt4.5':
        answer = query_gpt45_with_image(image_url, prompt)
    else:   
        answer = query_mFM_with_image(prompt, image_url, FM_model)
    
    try:
        answer = ast.literal_eval(answer.strip())
        if isinstance(answer, int):
            answer = [answer]
        elif isinstance(answer, float):
            answer = [int(answer)]
    except:
        answer = gpt4_verify_answer_format(answer)
    return answer


# Image-level Grounding.
def iterate_image_instructions(image_url, instructions, FM_model, type='', specialist=False):
    print(f"{image_url} & {type}: {len(instructions)} instructions to be processed\n")
    identified_list, missing_ids = [],[]
    # Generalist FMs
    if not specialist:
        for instruct in instructions:    
            user_prompts = ''.join(instruct)
            try:
                answer = instruction_grounding(image_url, FM_model, user_prompts) 
                identified_list += answer
            except Exception as e:
                print(f"Error: Missing data for image {img_id}. Exception raised {e}.")
                missing_ids.append(img_id)
                continue
            time.sleep(4)
        return identified_list, missing_ids
    # Specialist FMs
    else:
        for i, instruct in enumerate(instructions):   
            img_id = os.path.splitext(os.path.basename(image_url))[0]
            ref_img_path = f"output_dinoref/{img_id}_{type}_{i}_ref.png"
            if os.path.exists(ref_img_path):
                continue

            # llm for parsing instructions 
            parsed_dict = llm_parse_instruction(instruct)

            # detector to detect related objects
            obj_list = [parsed_dict['target']]
            if 'neighbors' in parsed_dict.keys() and parsed_dict['neighbors'] != "":
                obj_list.append(parsed_dict['neighbors'])

            candidate_img = f"output_dinoref/{img_id}_{type}_{i}_candidates.png"
            det_result = GroundingDINO(imageNum_url, obj_list) 
            if not os.path.exists(candidate_img):
                draw_candidate_boxes(imageNum_url, det_result, save_file=candidate_img)
            
            # det_result should not be empty
            if det_result['boxes'].shape[0] == 0:
                print(f"No object detected in the image!  {det_result}")
                identified_list += [0]
            # select target  from the detected objects
            else:
                target = llm_select_target(instruct, det_result)
                if not isinstance(target, dict):
                    print(f"Error: {image_url}, {instructions}: LLM select target result is not a dictionary!  {target}")
                if target == {}:
                    identified_list += [0]
                else:
                    # mllm for extracting the refnum.
                    draw_candidate_boxes(imageNum_url, target, save_file=ref_img_path)
                    draw_overlay_caption(ref_img_path, instruct)
                    try:
                        ref_answer = mllm_identify_obj_num(ref_img_path)
                        identified_list += ref_answer
                    except:
                        print(f"Error: Missing data for image {img_id}.")
                        missing_ids.append(img_id)
                        continue

        return identified_list, missing_ids



if __name__ == "__main__":
    # Load Data
    data = load_annotations("data_ref/instruction_gpt4_all.json")
    easy_ids, medium_ids, hard_ids = get_easy_medium_hard_items(data)
    data_grps = {'easy': easy_ids, 'medium': medium_ids, 'hard': hard_ids}

    # Set Foundation Models 
    # Specialists
    Specialists = ["GroundingDINO"] # set global var 

    # Generalists
    Azure_FMs = ["gpt4vision", "gpt-4o-mini-20240718",  "gpt4o0513", "phi-3.5"]  
    Together_FMs = ["meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", 
                    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
                    "Qwen/Qwen2-VL-72B-Instruct"]
    ollama_FMs = ["llama3.2-vision", "llama3.2-vision:90b"]
    deepseek_FMs = ["Janus-Pro-7B"]
    Aliyun_FMs = ["Qwen2.5-VL-72B"]
    Gemini_FMs = ["Gemini2.5Pro", "Gemini2.0Flash"]

    # Testing 
    FM_testing = Gemini_FMs[0]
    FM_savefile = Gemini_FMs[0]
    FM_oformat = Azure_FMs[0] 
    print(f"Testing Foundation Model: {FM_testing} \n")

    # Initialize API via Azure or Together
    if FM_testing in ["gpt4vision", "gpt-4o-mini-20240718"]:
        client = AzureOpenAI(
            api_key = "add your key here",  
            api_version = "2024-02-15-preview",
            azure_endpoint="add your endpoint here.",
            )
    elif FM_testing in ["gpt4o0513"]:
        client = AzureOpenAI(
            api_key= "add your key here",
            api_version = "2024-08-01-preview",
            azure_endpoint="add your endpoint"
            )
    elif FM_testing in ["phi-3.5"]:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.credentials import AzureKeyCredential

        keys = { 'llama': 'add your key here',
                'phi35': 'add your key here',
                'gpt4o': 'add your key here' }

        model_names = { 'llama': 'Llama-3.2-90B-Vision-Instruct',
                        'phi35': 'Phi-3.5-vision-instruct',
                        'gpt4o': 'gpt-4o' }

        endpoints = { 'llama': 'add ur endpoint',
                    'phi35': 'add your endpoint,
                    'gpt4o': 'https://[add your endpoint username].openai.azure.com/openai/deployments/gpt-4o' }
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
        os.environ['TOGETHER_API_KEY'] = "add your key here"
        client = Together()
    elif FM_testing in ollama_FMs:
        print("Testing Ollama Foundation Models")
    elif FM_testing in deepseek_FMs:
        print("Testing Deepseek Janus Pro 7B Foundation Model")
    elif FM_testing in Gemini_FMs:
        print("Testing Gemini Foundation Models")

    # Goal Specification Test
    results = defaultdict(dict)

    for difficulty_level in ['easy', 'medium', 'hard']:
        print(f"-------------{difficulty_level}: {len(data_grps[difficulty_level])} images ------\n")
        # initialize results
        macro_acc =  {'implicit': 0.0, 'attr': 0.0, 'rel': 0.0}
        macro_counter = {'implicit': 0, 'attr': 0, 'rel': 0}
        missing_ids = {'implicit': [], 'attr': [], 'rel': []}

        grounding = defaultdict(lambda: defaultdict(dict))

        # iterate subsets
        for img_id in data_grps[difficulty_level]:
            if FM_testing == 'GroundingDino':
                check_path = f"output_dinoref/{img_id[:-4]}_implicit_0_candidates.png"
                if os.path.exists(check_path):
                    continue

            if FM_testing in ["llama3.2-vision", "llama3.2-vision:90b","Janus-Pro-7B", "GroundingDINO", "Gemini2.5Pro", "Gemini2.0Flash"]:
                imageNum_url = f"data_ref/Images_NumPrompts/{img_id}"
            else:
                imageNum_url = local_image_to_data_url(f"data_ref/Images_NumPrompts/{img_id}") 
            print(f"Testing {img_id} \n")

            # Grounding Instructions
            ann_image_instructions = {
                'implicit': data[img_id]['implicit_instructions'],
                'attr': data[img_id]['explicit_attr_instructions'],
                'rel':  data[img_id]['explicit_rel_instructions']
                }   
            for instruction_type, ann_instructions in ann_image_instructions.items():
                if len(ann_instructions) == 0:
                    continue
                else:
                    print(f"Multiple instructions: {ann_instructions}")
                    grd_list = [int(ann) for ann in ann_instructions.keys()]
                    instruction_list = list(ann_instructions.values())
                    macro_counter[instruction_type] += 1
                
                print(f"Instruction Type: {instruction_type}")
                if FM_testing in Specialists:
                    identified_list, missing_list = iterate_image_instructions(imageNum_url, instruction_list, FM_testing, type=instruction_type,specialist=True)
                else:
                    identified_list, missing_list= iterate_image_instructions(imageNum_url, instruction_list, FM_testing)
                    #breakpoint()

                flag_list = [1 if a == b else 0 for a, b in zip(grd_list, identified_list)]
                grounding[img_id][instruction_type]['instruction'] = instruction_list
                grounding[img_id][instruction_type]['grd_truth'] =  grd_list
                grounding[img_id][instruction_type]['identified'] = identified_list 
                grounding[img_id][instruction_type]['flags'] = flag_list
                if len(flag_list) == 0:
                    grounding[img_id][instruction_type]['acc'] = 0.0
                else:
                    grounding[img_id][instruction_type]['acc'] = sum(flag_list) / len(flag_list)
                macro_acc[instruction_type] += grounding[img_id][instruction_type]['acc']
                missing_ids[instruction_type] = missing_list

        # Calculate Macro Accuracy
        results[difficulty_level] = grounding
        results['missing_ids'] = missing_ids
        macro_acc = {key: macro_acc[key] / macro_counter[key] for key in macro_acc}
        results[difficulty_level]['macro_acc'] = macro_acc
        # pprint.pprint(grounding) 
        # pprint.pprint(macro_acc)
        print(f"Macro Accuracy: {results[difficulty_level]['macro_acc']}\n")
        
        # Save results to a JSON file
        with open(f"{FM_savefile}_ref_results.json", "w") as json_file:
            json.dump(results, json_file, indent=4)
    

    table_results = defaultdict(dict)
    for difficulty_level in ['easy', 'medium', 'hard']:
        table_results[difficulty_level] = results[difficulty_level]['macro_acc']

    with open(f"{FM_savefile}_ref_table_results.json", "w") as json_file:
        json.dump(table_results, json_file, indent=4)
    print(f"Grounding data has been saved to json files.")
