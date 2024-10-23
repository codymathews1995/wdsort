import argparse
import os
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image, UnidentifiedImageError
import logging
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Path to the local model and labels
MODEL_PATH = "model/model.onnx"
LABEL_FILENAME = "model/selected_tags.csv"

kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger CLI")
    parser.add_argument("--folder", type=str, help="Path to the folder containing images or videos.")
    parser.add_argument("--scan", type=str, help="Path to a single image to scan for tags.")
    parser.add_argument("--bytag", type=str, help="Filter tags by specified tag.")
    parser.add_argument("--general-thresh", type=float, default=0.35, help="General tags threshold.")
    parser.add_argument("--character-thresh", type=float, default=0.85, help="Character tags threshold.")
    parser.add_argument("--mcut-general", action="store_true", help="Use MCut threshold for general tags.")
    parser.add_argument("--mcut-character", action="store_true", help="Use MCut threshold for character tags.")
    return parser.parse_args()

def load_labels(dataframe) -> tuple:
    name_series = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in kaomojis else x)
    tag_names = name_series.tolist()

    rating_indexes = np.where(dataframe["category"] == 9)[0].tolist()
    general_indexes = np.where(dataframe["category"] == 0)[0].tolist()
    character_indexes = np.where(dataframe["category"] == 4)[0].tolist()
    
    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs):
    sorted_probs = np.sort(probs)[::-1]
    diffs = np.diff(sorted_probs)
    t = diffs.argmax()
    return (sorted_probs[t] + sorted_probs[t + 1]) / 2

class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.load_model()

    def load_model(self):
        tags_df = pd.read_csv(LABEL_FILENAME)
        self.tag_names, self.rating_indexes, self.general_indexes, self.character_indexes = load_labels(tags_df)

        self.model = rt.InferenceSession(MODEL_PATH)
        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

    def prepare_image(self, image):
        try:
            if isinstance(image, str):
                logger.info(f"Attempting to open image: {image}")
                image = Image.open(image).convert("RGBA")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGBA")

            max_dim = max(image.size)
            target_size = self.model_target_size

            padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded_image.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))

            if max_dim != target_size:
                padded_image = padded_image.resize((target_size, target_size), Image.LANCZOS)

            return np.expand_dims(np.array(padded_image, dtype=np.float32)[:, :, ::-1], axis=0)

        except UnidentifiedImageError as e:
            logger.warning(f"Could not identify image file {image}: {e}")
            return None

    def predict(self, image_path, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
        image = self.prepare_image(image_path)
        if image is None:
            logger.warning(f"Skipping prediction for {image_path} due to loading error.")
            return []

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        general_names = self.get_filtered_tags(labels, self.general_indexes, general_thresh, general_mcut_enabled)
        character_names = self.get_filtered_tags(labels, self.character_indexes, character_thresh, character_mcut_enabled)

        return sorted(character_names + general_names, key=lambda x: x[1], reverse=True)

    def get_filtered_tags(self, labels, indexes, threshold, mcut_enabled):
        names = [labels[i] for i in indexes]
        if mcut_enabled:
            probs = np.array([x[1] for x in names])
            threshold = max(0.15, mcut_threshold(probs))
        
        return [(x[0], x[1]) for x in names if x[1] > threshold]

def process_single_image(predictor, image_path, args):
    tags = predictor.predict(image_path, args.general_thresh, args.mcut_general, args.character_thresh, args.mcut_character)
    
    if args.bytag:
        tags = [tag for tag in tags if args.bytag.lower() in tag[0].lower()]
    
    if args.scan:
        formatted_tags = "\n".join([f"- {tag[0]}: {tag[1]:.2f}" for tag in tags]) if tags else "No tags found."
        logger.info(f"Tags for {image_path}:\n{formatted_tags}")
    
    return tags

def move_image_to_folder(image_path, tags):
    if tags:
        first_tag = tags[0][0].replace(':', '-')  # Replace ":" with "-"
        tag_folder = os.path.join(os.path.dirname(image_path), first_tag)

        try:
            os.makedirs(tag_folder, exist_ok=True)
            new_image_path = os.path.join(tag_folder, os.path.basename(image_path))
            os.rename(image_path, new_image_path)
            logger.info(f"Moved {os.path.basename(image_path)} to {tag_folder}")
        except OSError as e:
            logger.error(f"Skipping folder creation for '{tag_folder}': {e}")

def process_folder_images(predictor, folder_path, args):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Process images
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            tags = process_single_image(predictor, file_path, args)

            if args.bytag:
                tags = [tag for tag in tags if args.bytag.lower() in tag[0].lower()]

            if tags:
                move_image_to_folder(file_path, tags)
            else:
                logger.info(f"No matching tags for {filename} with filter '{args.bytag}'. Skipping move.")
        
        # Process videos
        elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            process_video(predictor, file_path, args)

def process_video(predictor, video_path, args):
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the middle frame index
    middle_frame_index = total_frames // 2
    
    # Set the video position to the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    # Read the middle frame
    ret, frame = cap.read()
    cap.release()  # Release the video capture
    
    if not ret:
        logger.warning(f"Could not read frame {middle_frame_index} from {video_path}.")
        return
    
    # Process the middle frame
    tags = predictor.predict(frame, args.general_thresh, args.mcut_general, args.character_thresh, args.mcut_character)
    
    if args.bytag:
        tags = [tag for tag in tags if args.bytag.lower() in tag[0].lower()]

    if tags:
        logger.info(f"Tags for the middle frame of {video_path}:\n" +
                    "\n".join([f"- {tag[0]}: {tag[1]:.2f}" for tag in tags]))

        # Move the video file based on the tags
        first_tag = tags[0][0].replace(':', '-')  # Replace ":" with "-"
        tag_folder = os.path.join(os.path.dirname(video_path), first_tag)

        try:
            os.makedirs(tag_folder, exist_ok=True)
            new_video_path = os.path.join(tag_folder, os.path.basename(video_path))
            os.rename(video_path, new_video_path)
            logger.info(f"Moved {os.path.basename(video_path)} to {tag_folder}")
        except OSError as e:
            logger.error(f"Skipping folder creation for '{tag_folder}': {e}")
    else:
        logger.info(f"No matching tags for the middle frame of {video_path} with filter '{args.bytag}'.")

def main():
    args = parse_args()
    predictor = Predictor()

    if args.scan:
        process_single_image(predictor, args.scan, args)
        return

    if args.folder:
        process_folder_images(predictor, args.folder, args)

if __name__ == "__main__":
    main()
