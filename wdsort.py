import argparse
import os
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image
import logging

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
    parser.add_argument("--folder", type=str, help="Path to the folder containing images.")
    parser.add_argument("--scan", type=str, help="Path to a single image to scan for tags.")
    parser.add_argument("--general-thresh", type=float, default=0.35, help="General tags threshold.")
    parser.add_argument("--character-thresh", type=float, default=0.85, help="Character tags threshold.")
    parser.add_argument("--mcut-general", action="store_true", help="Use MCut threshold for general tags.")
    parser.add_argument("--mcut-character", action="store_true", help="Use MCut threshold for character tags.")
    parser.add_argument("--bytag", type=str, help="Filter tags by specified tag.")
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

    def prepare_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGBA")
        except OSError as e:
            logger.error(f"Error opening image {image_path}: {e}")
            return None
        
        max_dim = max(image.size)
        target_size = self.model_target_size

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))

        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.LANCZOS)

        return np.expand_dims(np.array(padded_image, dtype=np.float32)[:, :, ::-1], axis=0)

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
        tags = [tag for tag in tags if args.bytag in tag]
    
    # Log tags only when scanning a single image
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
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            image_path = os.path.join(folder_path, filename)
            tags = process_single_image(predictor, image_path, args)

            # Filter tags if bytag is specified
            if args.bytag:
                tags = [tag for tag in tags if args.bytag in tag]

            # Only move the image if there's a matching tag
            if tags:
                move_image_to_folder(image_path, tags)
            else:
                logger.info(f"No matching tags for {filename} with filter '{args.bytag}'. Skipping move.")

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
