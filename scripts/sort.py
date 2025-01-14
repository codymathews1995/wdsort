import os
from pathlib import Path
from PIL import Image
import shutil
import mimetypes
import cv2

def get_orientation(image_path):
    """Determine if an image or gif is portrait, landscape, or square."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width == height:
                return "Square"
            return "Portrait" if height > width else "Landscape"
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "Unknown"

def get_video_orientation(video_path):
    """Determine the orientation of a video based on its aspect ratio."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Cannot open video file")
        
        # Read the first frame
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to read frame from video")
        
        height, width, _ = frame.shape
        
        cap.release()
        
        if width == height:
            return "Square"
        return "Portrait" if height > width else "Landscape"
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return "Unknown"

def get_file_type(file_path):
    """Determine if a file is an image, video, or gif."""
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type is None:
        return None
    
    if mime_type.startswith('image/'):
        if mime_type == 'image/gif':
            return 'Gif'
        return 'Image'
    elif mime_type.startswith('video/'):
        return 'Video'
    return None

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def sort_media_files(folder_path):
    # Validate directory exists
    if not os.path.isdir(folder_path):
        print("Invalid directory path!")
        return
    
    # Initialize mimetypes
    mimetypes.init()
    
    # Create main category directories
    categories = ['Image', 'Video', 'Gif']
    orientations = ['Portrait', 'Landscape', 'Square', 'Unknown']
    
    for category in categories:
        for orientation in orientations:
            create_directory(os.path.join(folder_path, category, orientation))
    
    # Process files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Skip if it's a directory
        if os.path.isdir(file_path):
            continue
        
        # Get file type
        file_type = get_file_type(file_path)
        if file_type is None:
            continue
        
        # Get orientation for images and gifs
        orientation = "Unknown"
        if file_type == 'Image' or file_type == 'Gif':
            orientation = get_orientation(file_path)
        elif file_type == 'Video':
            orientation = get_video_orientation(file_path)
        
        # Create destination path
        dest_path = os.path.join(folder_path, file_type, orientation, filename)
        
        # Move file
        try:
            shutil.move(file_path, dest_path)
            print(f"Moved {filename} to {file_type}/{orientation}/")
        except Exception as e:
            print(f"Error moving {filename}: {e}")