# From Language to Pixels: Task Recognition and Task Learning in LLMs

This repository contains the paper, code, data, and prompts used in the research published at the [**GenBench Workshop at EMNLP 2024**](https://aclanthology.org/2024.genbench-1.2/).

## Abstract  
LLMs can perform unseen tasks by learning from a few in-context examples. How in-context learning works is still uncertain. We investigate the mechanisms of in-context learning on a challenging non-language task. The task requires the LLM to generate pixel matrices representing images of basic shapes. We introduce a framework to analyze if this task is solved by recognizing similar formats from the training data (task recognition) or by understanding the instructions and learning the skill de novo during inference (task learning). Our experiments demonstrate that LLMs generate meaningful pixel matrices with task recognition and fail to learn such tasks when encountering unfamiliar formats. Our findings offer insights into LLMs’ learning mechanisms and their generalization ability to guide future research on their seemingly human-like behavior.

This repository supports our findings with all necessary materials for replication and further exploration.


## Repository Contents  

- **`paper/`**: The paper published at GenBench workshop.
- **`code/`**: Python scripts for running experiments and analyzing outputs.  
- **`prompts/`**: Instruction templates and example prompts used in experiments. 
- **`output/`**: Pixel matrix outputs generated by LLMs, categorized by task format and input variations.

---

## The python script
This Python script generates visual outputs (pixel or SVG-based images) based on prompts processed through selected language models, including OpenAI's GPT and various Hugging Face models. The results are saved as text and using PIL as images.

### Set Parameters: 
The parameters are all set inside the script. 
  - model_id: 
  - prompt_type: Choose "pixel" or "SVG".
  - use_GK_pixels: Use G/K instead of 0/1.
  - pixel_symbols: Define pixel colors/symbols.
  - prompt_path: the path to a txt file with a prompt
  - objects: a list of strings that describe the objects that should be generated
### Run the Script: 
python script.py
### Output
Text: Stored in output/text/

Images: Stored in output/images/
