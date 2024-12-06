
import os
import time
import openai
import requests
from PIL import Image
import cairosvg
import re

# models to choose from
bloom = "bigscience/bloom"
gpt_neo = "EleutherAI/gpt-neox-20b"
starcoder = "bigcode/starcoder"
codelama34 = "Phind/Phind-CodeLlama-34B-v2"
codelama = "codellama/CodeLlama-7b-hf"
codelama13 = "codellama/CodeLlama-13b-Instruct-hf"
deepseek = "deepseek-ai/deepseek-coder-33b-instruct"
falcon = "tiiuae/falcon-7b-instruct"
metaLama = "meta-llama/Llama-2-7b-chat-hf"
openAI = "gpt-3.5-turbo"#"gpt-4"" 

# choose the model to use here
model_id = openAI

# promtp type can be "pixel" or "svg"
prompt_type = "pixel"

# true if G and K are used instead of 0 and 1 inside the pixel matrix
use_GK_pixels = False

# true if a self-generated visual description of the object should be used
use_generated_description = False

# pixel symbols is used to determine which symbols are used in the pixel matrix and thus create the image
# pixel_symbols = ["0", "1", "2", "3", "4", "5"] # colors
pixel_symbols = ["0", "1"] # black and white
# pixel_symbols = ["G", "K"] # black and white with G and K instead of 0 and 1
# pixel_symbols = "RGB" # RGB colors

prompt_path = 'Prompts/Experiment_baseline/pixel_letters.txt'
number_of_completions = 10

# api parameters
openai.api_key = "xxx"
API_URL = "https://api-inference.huggingface.co/models/{}".format(model_id)
headers = {"Authorization": "xxx"}

# query the chosen model from huggingface
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def main():
    #objects = ["number 0", "a number 1", "a number 2", "a number 3", "a number 4", "a number 5", "a number 6", "a number 7", "a number 8", "the number 9", "the number 10", "the number 32"]
    # objects = ["the capital letter A", "the capital letter B", "the capital letter C", "the capital letter D", "the capital letter E", "the capital letter F", "the capital letter G", "the capital letter H", "the capital letter I", "the capital letter J", "the capital letter K", "the capital letter L", "the capital letter M", "the capital letter N", "the capital letter O", "the capital letter P", "the capital letter Q", "the capital letter R", "the capital letter S", "the capital letter T", "the capital letter U", "the capital letter V", "the capital letter W", "the capital letter X", "the capital letter Y", "the capital letter Z", "the letter Ä", "the letter Ö", "the letter Ü", "the letter ß"]
    # objects = ["the symbol of a(n) comma", "the symbol of a(n) semicolon", "the symbol of a(n) exclamation point", "the symbol of a(n) equal sign", "the symbol of a(n) plus sign", "the symbol of a(n) hashtag", "the symbol of a(n) dollar sign", "the symbol of a(n) percent sign", "the symbol of a(n) ampersand", "the symbol of a(n) asterisk", "the symbol of a(n) left parenthesis", "the symbol of a(n) right parenthesis", "the symbol of a(n) left bracket", "the symbol of a(n) right bracket", "the symbol of a(n) left curly brace", "the symbol of a(n) right curly brace", "the symbol of a(n) less than sign", "the symbol of a(n) greater than sign", "the symbol of a(n) back slash", "the symbol of a(n) underscore", "the symbol of a(n) colon", "the symbol of a(n) single quote", "the symbol of a(n) double quote", "the symbol of a(n) at sign", "the symbol of a(n) caret"]
    # objects = ["sad face", "cup", "heart", "wine glass half full", "cactus", "key", "skull", "mouse", "crown", "lightning flash", "padlock",  "cat",  "crab",  "a chess board", "a house", "coffee", "car", "window", "chair", "star", "mountain", "sun", "boat", "stick figure", "fly"]
    for i in range(len(objects)):
        object_description = "" + objects[i]
        experiment(object_description)
        time.sleep(10)

# run one experiment with the given parameters, preparing prompts and saving the results
def experiment(object_description):
    with open(prompt_path) as f:
        prompt = f.read().replace("[object]", object_description)

    # prepare the object description for the file name
    object_description_prepared = object_description.replace(" ", "_")
    model_id_prepared = model_id.replace("/", "_")

    # pimp prompt by adding a generated visual description of the object
    if use_generated_description: 
        with open("Prompts/Experiment_textual_description/world_objects_2.txt") as f:
            visual_description_prompt = f.read().replace("[object]", object_description)
        # generate a visual description of the object
        completion = openai.ChatCompletion.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": visual_description_prompt}
                ],
                n=1,
                temperature=0.9,
            )
        generated_contents = [choice["message"]["content"] for choice in completion["choices"]]
        # add the generated visual description to the prompt
        with open("output/text/{model}_{prompt_type}_{obj}_00.txt".format(obj=object_description_prepared, prompt_type = prompt_type, model = model_id_prepared), "w") as f:
            f.write(generated_contents[0])
        prompt = prompt.replace("[description]", "The object to visualize can be described as follows: "+"\""+generated_contents[0] + "\"")

    # query the chosen model
    if model_id != openAI:
        query_response = query({
            "inputs": prompt,
            "paramters": {
                'max_new_tokens': 100,
                'temperature': 1,
                'return_full_text': False,
            },
        })
        # generated_contents = [query_response[0]["generated_text"]]
        print(query_response)
        generated_contents = [query_response_i["generated_text"].split("###")[0] for query_response_i in query_response]
    else:
        completion = openai.ChatCompletion.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                n=number_of_completions,
                temperature=1,
                stop=["###"],
            )
        generated_contents = [choice["message"]["content"] for choice in completion["choices"]]
    

   
    # save each generated content to a text and an image file
    for i in range(len(generated_contents)):
      with open("output/text/{model}_{prompt_type}_{obj}_{i}.txt".format(obj=object_description_prepared, prompt_type = prompt_type, model = model_id_prepared, i = i), "w") as f:
        f.write(generated_contents[i])
      if prompt_type == "pixel":
        if pixel_symbols == "RGB": 
            image = convert_to_image_RGB(generated_contents[i])
        else: image = convert_to_image(generated_contents[i])
        image.save("output/images/{model}_{prompt_type}_{obj}_{i}.png".format(obj=object_description_prepared, prompt_type = prompt_type, model = model_id_prepared, i = i))
      elif prompt_type == "svg":
        svg_string = generated_contents[i]
        try: 
          cairosvg.svg2png(bytestring=svg_string.encode('utf-8'), output_height=400, output_width=400, write_to="output/images/{model}_{prompt_type}_{obj}_{i}.png".format(obj=object_description_prepared, prompt_type = prompt_type, model = model_id_prepared, i = i))
        except Exception as e: 
          print("error saving image {model}_{prompt_type}_{obj}_{i}.png".format(obj=object_description_prepared, prompt_type = prompt_type, model = model_id_prepared, i = i)+ str(e)) 


# convert the generated content to an image         
def convert_to_image(generated_content):

    # split the string into lines and filter out all lines that do not contain any pixel symbols
    response_split_filtered = filter_response(generated_content.splitlines())

    #find the string with the most characters in the response_split_filtered list to determine the number of columns
    longest_string = ""
    for i in range(len(response_split_filtered)):
        if i == 0:
            longest_string = response_split_filtered[i]
        elif len(response_split_filtered[i]) > len(longest_string):
            longest_string = response_split_filtered[i]
    nr_rows = len(response_split_filtered)
    nr_columns = len(longest_string)

    # if nr rows or nr columns is 0 return an empty image
    if nr_rows == 0 or nr_columns == 0:
        return Image.new("RGB", (1, 1))
    
    pixelMatrix = [ [ 0 for _ in range(nr_columns) ] for _ in range(nr_rows) ]
    for i, line in enumerate(response_split_filtered): 
        for j, char in enumerate(line):
            pixelMatrix[i][j]= char 

    image = Image.new("RGB", (nr_columns, nr_rows))
    for i in range(nr_rows):
        for j in range(nr_columns):
            if pixelMatrix[i][j] == "0":
                image.putpixel((j, i), (255, 255, 255))
            elif pixelMatrix[i][j] == "1":
                image.putpixel((j, i), (0, 0, 0))
            elif pixelMatrix[i][j] == "2":
                image.putpixel((j, i), (255, 0, 0))
            elif pixelMatrix[i][j] == "3":
                image.putpixel((j, i), (255, 255, 0))
            elif pixelMatrix[i][j] == "4":
                image.putpixel((j, i), (0, 255, 0))
            elif pixelMatrix[i][j] == "5":
                image.putpixel((j, i), (0, 0, 255))
            elif pixelMatrix[i][j] == "G":
                image.putpixel((j, i), (255, 255, 255))
            elif pixelMatrix[i][j] == "K":
                image.putpixel((j, i), (0, 0, 0))
            else: 
                image.putpixel((j, i), 	(255,0,255))
                
    return image.resize((nr_columns*50, nr_rows*50), resample=0)

# convert the generated RGB content to an image
def convert_to_image_RGB(generated_content):

    # split the string into lines, no filter here for COT responses
    lines = generated_content.strip('[ ]').splitlines()
    
    # remove lines that do not contain any digits
    lines = [line for line in lines if any(char.isdigit() for char in line)]

    # split each line into the pixel values
    for i in range(len(lines)):
        lines[i] = lines[i].strip('[ ]')
        lines[i] = re.split(r'\)\(|\)\s\(', lines[i])

    # Determine the number of rows and columns, check for longest if the completion has wrong format
    longest_line = ""
    for i in range(len(lines)):
        if i == 0:
            longest_line = lines[i]
        elif len(lines[i]) > len(longest_line):
            longest_line = lines[i]
    nr_rows = len(lines)
    nr_columns = len(longest_line)
    print("Rows: ", nr_rows)
    print("Columns: ", nr_columns)

    # if nr rows or nr columns is 0 return an empty image colored in magenta
    if nr_rows == 0 or nr_columns == 0:
        return Image.new("RGB", (1, 1), (255,0,255))

    image = Image.new("RGB", (nr_columns, nr_rows))
    for i in range(nr_rows):
        pixel_values = lines[i]
        for j in range(nr_columns):
            try:
                r, g, b = map(int, pixel_values[j].strip('()').split(','))
            except: 
                # check if j is out of range
                if j >= len(pixel_values): print("j is out of range: ", j)
                else: print("error converting pixel values to int: ", pixel_values[j].strip('()').split(','))
                r, g, b = 255, 0, 255
            image.putpixel((j, i), (r, g, b))

    return image.resize((nr_columns*50, nr_rows*50), resample=0)
    
#filter everything that is not one row of an image (needed for cot responses)
def filter_response(response_list):
    filtered_response = []
    # for all lines in the response_list filter out all symbols that are not part of the pixel symbols 
    for line in response_list:
        filtered_line = ""
        for char in line:
            if char in pixel_symbols:
                filtered_line += char
        if filtered_line != "":
            filtered_response.append(filtered_line)
    return filtered_response

if __name__ == "__main__":
    main()