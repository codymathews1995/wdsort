# wdsort

**wdsort** is a command-line interface (CLI) tool designed to help users sort images into folders using the WaifuDiffusion model. 

This program utilizes the model from [Hugging Face - SmilingWolf/wd-v1-4-swinv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2) to categorize images based on their content.

## Features

- **Image Sorting**: Automatically sort images into designated folders based on their content.
- **Easy to Use**: Simple CLI commands for easy access to users of any skill level.

## Prerequisites
Before using **wdsort**, ensure you have the following:
- Python 3.7 or higher
- Required Python libraries (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wdsort.git
   ```

2. Navigate to project directory:
   ```bash
   cd wdsort
   ```
3. Create virtual environment and activate:
   ```bash
   python -m venv .venv && .venv\Scripts\Activate
   ```
4. Install requirements
   ```bash
   pip install -r requirements.txt
   ```
5. Download model.onnx, selected_tags.csv, and config.json from https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/tree/main

6. Place these files in model directory

## Using the Program
  ```bash
  python appv2.py --folder "path/to/folder"
  ```
  This should sort images based on the first tag found with an inherent bias towards any character tags identified.

## Note
This is not the perfect solution for organizing images. I am new to programming in general, so there is still a lot of work to do. 
Primary credit goes to SmilingWolf for offering the model on HugggingFace open source.

## To Do
[] Add more customized sorting criteria
