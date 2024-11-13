# processors.py
import os
import cv2
from scripts.config import logger
from scripts.utils import move_to_folder

def process_image(predictor, image_path, args):
    tags = predictor.predict(
        image_path,
        args.general_thresh,
        args.mcut_general,
        args.character_thresh,
        args.mcut_character
    )
    
    if args.exclude:
        exclude_tags = [tag.lower() for tag in args.exclude]
        tags = [tag for tag in tags if tag[0].lower() not in exclude_tags]
    
    if args.bytag:
        tags = [tag for tag in tags if args.bytag.lower() in tag[0].lower()]
    
    if args.scan:
        formatted_tags = "\n".join([f"- {tag[0]}: {tag[1]:.2f}" for tag in tags]) if tags else "No tags found."
        logger.info(f"Tags for {image_path}:\n{formatted_tags}")
    
    return tags

def process_video(predictor, video_path, args):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_index = total_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.warning(f"Could not read frame {middle_frame_index} from {video_path}.")
        return
    
    tags = process_image(predictor, frame, args)
    
    if tags:
        logger.info(f"Tags for the middle frame of {video_path}:\n" +
                    "\n".join([f"- {tag[0]}: {tag[1]:.2f}" for tag in tags]))
        move_to_folder(video_path, tags, args.exclude if args.exclude else None)
    else:
        logger.info(f"No matching tags for the middle frame of {video_path} with filter '{args.bytag}'.")

def process_folder(predictor, folder_path, args):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            tags = process_image(predictor, file_path, args)
            if tags:
                move_to_folder(file_path, tags, args.exclude if args.exclude else None)
            else:
                logger.info(f"No matching tags for {filename} with filter '{args.bytag}'. Skipping move.")
        
        elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            process_video(predictor, file_path, args)
