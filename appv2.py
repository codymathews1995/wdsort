import argparse
import os
import numpy as np
import onnxruntime as rt
import pandas as pd
from PIL import Image

# Path to the local model and labels
MODEL_PATH = "model/model.onnx"
LABEL_FILENAME = "model/selected_tags.csv"

kaomojis = [
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=", ">_<",
    "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger CLI")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--general-thresh", type=float, default=0.35, help="General tags threshold.")
    parser.add_argument("--character-thresh", type=float, default=0.85, help="Character tags threshold.")
    parser.add_argument("--mcut-general", action="store_true", help="Use MCut threshold for general tags.")
    parser.add_argument("--mcut-character", action="store_true", help="Use MCut threshold for character tags.")
    return parser.parse_args()

def load_labels(dataframe) -> list[str]:
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes

def mcut_threshold(probs):
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

class Predictor:
    def __init__(self):
        self.model_target_size = None
        self.load_model()

    def load_model(self):
        csv_path = LABEL_FILENAME
        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        self.model = rt.InferenceSession(MODEL_PATH)
        _, height, width, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

    def prepare_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGBA")
        except OSError as e:
            print(f"Error opening image {image_path}: {e}")
            return None
        
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        if max_dim != target_size:
            padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)

        image_array = np.asarray(padded_image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(self, image_path, general_thresh, general_mcut_enabled, character_thresh, character_mcut_enabled):
        image = self.prepare_image(image_path)
        if image is None:  # If image loading failed
            print(f"Skipping prediction for {image_path} due to loading error.")
            return []  # Return an empty list or any other appropriate response
        
        
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        general_names = [labels[i] for i in self.general_indexes]
        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        character_names = [labels[i] for i in self.character_indexes]
        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_character_strings = sorted(character_res.items(), key=lambda x: x[1], reverse=True)
        sorted_character_strings = [x[0] for x in sorted_character_strings]

        sorted_general_strings = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
        sorted_general_strings = [x[0] for x in sorted_general_strings]

        final_tags = sorted_character_strings + sorted_general_strings
        return final_tags

def main():
    args = parse_args()

    predictor = Predictor()

    for filename in os.listdir(args.folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
            image_path = os.path.join(args.folder, filename)
            tags = predictor.predict(
                image_path,
                args.general_thresh,
                args.mcut_general,
                args.character_thresh,
                args.mcut_character,
            )

            if tags:
                first_tag = tags[0].replace(':', '-')  # Replace ":" with "-"
                tag_folder = os.path.join(args.folder, first_tag)

                # Create folder if it doesn't exist
                os.makedirs(tag_folder, exist_ok=True)

                # Move the image to the new folder
                new_image_path = os.path.join(tag_folder, filename)
                os.rename(image_path, new_image_path)
                print(f"Moved {filename} to {tag_folder}")

if __name__ == "__main__":
    main()
