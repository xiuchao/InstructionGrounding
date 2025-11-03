import os
import json
import pprint
import ast
from collections import defaultdict
import numpy as np
import torch
import requests
from PIL import Image 
import base64
from mimetypes import guess_type 

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageFont


# Dataset Preparation
def load_annotations(json_path):
    # Load the data from the file
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Data loaded successfully, {len(data)} images annotated.")
    return data 

def histogram(numbers):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import statistics

    sns.set_style("whitegrid")

    bins = [5, 10, 15, 20, 26]  # Define the bins
    plt.figure(figsize=(8, 6))  #
    hist_data = plt.hist(numbers, bins=bins, color="skyblue", edgecolor="black", alpha=0.7, rwidth=0.85)    

    # labels
    plt.xlabel('Number of Objects')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Objects')

    plt.xticks(ticks=[5, 10, 15, 20, 25], fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7) 

    for i in range(len(hist_data[0])):
        plt.text(hist_data[1][i] + 1.25, hist_data[0][i] + 0.1, str(int(hist_data[0][i])), ha='center')


    # annotation for the mean value
    mean_value = sum(numbers) / len(numbers)
    plt.axvline(mean_value, color='red', linestyle='--', label=f'Mean = {mean_value:.2f}')
    plt.legend(fontsize=12)
    plt.tight_layout()  
    plt.show()
    plt.close()
    return statistics.mean(numbers), statistics.stdev(numbers)

def get_object_distribution(data):
    counter = []  
    for img_id in data.keys():   
        counter.append(len(data[img_id]['objects']))
    counter_avg, counter_std= histogram(counter)
    print(f"Avg: {counter_avg :.2f}, Std: {counter_std:.2f}") 
    return counter_avg, counter_std

def get_easy_medium_hard_items(data):
    # Review objects distribution in the dataset
    if False:
        get_object_distribution(data)

    # Get the easy, medium, and hard items based on the number of objects in the image
    easy_items = []
    medium_items = []
    hard_items = []
    for img_id in data.keys():
        num_objects = len(data[img_id]['objects'])
        if num_objects < 10:
            easy_items.append(img_id)
        elif num_objects >= 10 and num_objects < 20:
            medium_items.append(img_id)
        else:
            hard_items.append(img_id)
    return easy_items, medium_items, hard_items

def local_image_to_data_url(image_path):
    # Encode a local image into data URL 
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


# unique item list for groundingdino
def get_unique_items(label_dict):
    unique_values = []
    for value in label_dict.values():
        if value not in unique_values:
            unique_values.append(value)

    list_of_strings = [unique_values]
    print(list_of_strings)
    return list_of_strings

# formatting the string data 
def preprocess_tensor_string(s):
    import re
    # extract the dictionary part using regex
    match = re.search(r"(\{.*\})", s, re.DOTALL)
    if match:
        dict_str = match.group(1)
    # Remove device=... inside tensor(...)
    s = re.sub(r"tensor\(([^()]+?),\s*device='[^']+'\)", r"\1", dict_str)
    # Remove tensor(...) wrapper
    s = re.sub(r"tensor\(([^()]+?)\)", r"\1", s)
    return s

def parse_tensor_string(s):
    import ast
    cleaned = preprocess_tensor_string(s)
    try:
        result_dict = ast.literal_eval(cleaned)
    except:
        print(cleaned)
        breakpoint()
    return result_dict

def string_of_empty_dict(s):
    try:
        return isinstance(ast.literal_eval(s), dict) and not ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return False

# Plot detection results
def draw_candidate_boxes(image_pil, result, save_file=''):
    if isinstance(image_pil, str):
        filename = image_pil
        image_pil = Image.open(image_pil).convert("RGB")
    #image_pil = Image.open(image_path).convert("RGB") 
    #image_pil = image_pil.convert("RGB")
    H, W = image_pil.size
    draw = ImageDraw.Draw(image_pil)
    #mask = Image.new("L", image_pil.size, 0)  #box mask
    #mask_draw = ImageDraw.Draw(mask)

    try:    
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
            # print(f"Detected {label} with confidence {round(score.item(), 3)} at location {box}")
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            # draw rectangle
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=8)

            # draw textbox+text.
            fontPath = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
            sans16  =  ImageFont.truetype ( fontPath, 18 )
            font = sans16  #font = ImageFont.load_default()
            #label_txt = label[0]+"("+str(label[1])+")"
            label_txt = label
            if hasattr(font, "getbbox"):
                txtbox = draw.textbbox((x0, y0), label_txt, font)
            else:
                w, h = draw.textsize(label_txt, font)
                txtbox = (x0, y0, w + x0, y0 + h)

            draw.rectangle(txtbox, fill=color)
            draw.text((x0, y0), label_txt, fill="white", font=font)
            #mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=8)
    except:
        # breakpoint()
        print("Error in drawing boxes {filename}")
        print(result)

        
    if save_file:
        image_pil.save(save_file)
    
    return image_pil

def draw_overlay_caption(image_url, instruction):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.imshow(Image.open(image_url))
    plt.title(instruction)
    plt.axis('off')
    save_file = os.path.join(os.path.dirname(image_url), os.path.basename(image_url).split(".")[0] + "_caption.png")
    plt.savefig(save_file,  bbox_inches="tight", dpi=300, pad_inches=0.0)
    return True

if __name__ == "__main__":
    # Load Data
    data = load_annotations("data_specification/instruction_gpt4_all.json")
    easy_ids, medium_ids, hard_ids = get_easy_medium_hard_items(data)