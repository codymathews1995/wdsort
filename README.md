# wdsort

**wdsort** is a command-line interface (CLI) tool designed to help users sort images into folders using the WaifuDiffusion model.

This program utilizes the model from [Hugging Face - SmilingWolf/wd-v1-4-swinv2-tagger-v2](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2) to categorize images based on their content.

## Features

- **Image Sorting**: Automatically sort images into designated folders based on their content.
- **Easy to Use**: Simple CLI commands for easy access to users of any skill level.

## Prerequisites

Before using **wdsort**, ensure you have the following:

- Python 3.7 or higher
- Poetry installed to manage dependencies

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/wdsort.git
   ```

2. Navigate to the project directory:

   ```bash
   cd wdsort
   ```

3. Install dependencies using Poetry:

   If you don't have Poetry installed, follow the installation instructions [here](https://python-poetry.org/docs/#installation).

   Once Poetry is installed, run the following command to install the project dependencies:

   ```bash
   poetry install
   ```

4. Download `model.onnx`, `selected_tags.csv`, and `config.json` from [Hugging Face](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2/tree/main).

5. Place these files in the `model` directory.

## Using the Program

First, **spawn a shell**:

   ```bash
   poetry shell
   ```

Once in the shell, you can:

1. **To Scan by Folder**

   With a bias towards the first character identified. Otherwise, the first tag found.

   ```bash
   python wdsort.py --folder "path/to/folder"
   ```

2. **To Assess Tags of Single Image**

   Output all tags for a single image to the console.

   ```bash
   python wdsort.py --scan "path/to/image"
   ```

3. **Sort By Specific Tag**

   Will sort by a specified "tag" and any combination where that tag is used (e.g., "eat" and "eating").

   ```bash
   python wdsort.py --bytag "tag"
   ```

4. **Exclude tag(s)

   Will exclude the tags specified in conjunction with --folder tags

   ```bash
   python wdsort.py --folder "path/to/folder" --exclude "tag1" "tag2"
   ```

To exit the shell, type:

   ```bash
   exit
   ```

## Note

This is not the perfect solution for organizing images. I am new to programming in general, so there is still a lot of work to do. Primary credit goes to SmilingWolf for offering the model on HuggingFace.

If this program provides any legal issues to anybody, please let me know and I will take it down promptly.

## To Do

- [ ] Add more customized sorting criteria
