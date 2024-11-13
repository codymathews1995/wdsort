# utils.py
import os
import re
import shutil
from scripts.config import logger

def move_to_folder(source_path, tags, exclude_tags=None):
    """Move file to a folder based on its top tag."""
    if not tags:
        return
    
    if exclude_tags:
        tags = [tag for tag in tags if tag[0].lower() not in exclude_tags]
    
    if tags:
        first_tag = tags[0][0].replace(':', '-')
        tag_folder = os.path.join(os.path.dirname(source_path), first_tag)
        
        try:
            os.makedirs(tag_folder, exist_ok=True)
            new_path = os.path.join(tag_folder, os.path.basename(source_path))
            os.rename(source_path, new_path)
            folder_name = os.path.basename(tag_folder)
            logger.info(f"Moved {os.path.basename(source_path)} to folder '{folder_name}'")
        except OSError as e:
            logger.error(f"Skipping folder creation for '{tag_folder}': {e}")

def clean_folders(folder_path):
    """Organize directories based on names in parentheses."""
    if not os.path.isdir(folder_path):
        logger.error(f"The provided directory '{folder_path}' does not exist.")
        return

    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)
        if not os.path.isdir(folder_path_full):
            continue
            
        match = re.search(r'\s*\(([^)]+)\)$', folder_name)
        if not match:
            logger.info(f"No match for folder: '{folder_name}'. Skipping.")
            continue
            
        category = match.group(1).strip()
        original_name = re.sub(r'\s*\([^)]*\)$', '', folder_name).strip()
        
        category_folder = os.path.join(folder_path, category)
        os.makedirs(category_folder, exist_ok=True)
        
        new_folder_path = os.path.join(category_folder, original_name)
        
        if os.path.exists(new_folder_path):
            merge_folders(folder_path_full, new_folder_path)
        else:
            try:
                shutil.move(folder_path_full, new_folder_path)
                logger.info(f"Moved folder: '{folder_path_full}' to '{new_folder_path}'")
            except Exception as e:
                logger.error(f"Error moving '{folder_path_full}': {e}")

def merge_folders(source_folder, dest_folder):
    """Merge contents of source folder into destination folder."""
    logger.info(f"Folder '{dest_folder}' already exists. Merging contents.")
    
    for item in os.listdir(source_folder):
        source_item_path = os.path.join(source_folder, item)
        dest_item_path = os.path.join(dest_folder, item)
        
        try:
            if os.path.exists(dest_item_path):
                logger.info(f"File '{item}' already exists in '{dest_folder}'. Skipping.")
            else:
                shutil.move(source_item_path, dest_folder)
        except Exception as e:
            logger.error(f"Error moving '{source_item_path}' to '{dest_folder}': {e}")
    
    if not os.listdir(source_folder):
        os.rmdir(source_folder)
        logger.info(f"Removed empty folder: '{source_folder}'")
    else:
        logger.info(f"Folder '{source_folder}' not empty after merge.")