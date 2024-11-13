import numpy as np
import os
import onnxruntime as rt
import pandas as pd
from PIL import Image, UnidentifiedImageError
import cv2
from scripts.config import MODEL_PATH, LABEL_FILENAME, KAOMOJIS, logger

def load_labels(dataframe):
    name_series = dataframe["name"].map(lambda x: x.replace("_", " ") if x not in KAOMOJIS else x)
    tag_names = name_series.tolist()

    rating_indexes = np.where(dataframe["category"] == 9)[0].tolist()
    general_indexes = np.where(dataframe["category"] == 0)[0].tolist()
    character_indexes = np.where(dataframe["category"] == 4)[0].tolist()
    
    return tag_names, rating_indexes, general_indexes, character_indexes

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
                file_name = os.path.basename(image)
                logger.info(f"Attempting to open image: {file_name}")
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            image = image.convert("RGBA")
            max_dim = max(image.size)
            target_size = self.model_target_size

            padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded_image.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))

            if max_dim != target_size:
                padded_image = padded_image.resize((target_size, target_size), Image.LANCZOS)

            return np.expand_dims(np.array(padded_image, dtype=np.float32)[:, :, ::-1], axis=0)

        except (UnidentifiedImageError, ValueError) as e:
            logger.warning(f"Error processing image {image}: {e}")
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
        general_names = self.get_filtered_tags(labels, self.general_indexes, general_thresh, general_mcut_enabled)
        character_names = self.get_filtered_tags(labels, self.character_indexes, character_thresh, character_mcut_enabled)

        return sorted(character_names + general_names, key=lambda x: x[1], reverse=True)

    def get_filtered_tags(self, labels, indexes, threshold, mcut_enabled):
        names = [labels[i] for i in indexes]
        if mcut_enabled:
            probs = np.array([x[1] for x in names])
            threshold = max(0.15, self.mcut_threshold(probs))
        
        return [(x[0], x[1]) for x in names if x[1] > threshold]

    @staticmethod
    def mcut_threshold(probs):
        sorted_probs = np.sort(probs)[::-1]
        diffs = np.diff(sorted_probs)
        t = diffs.argmax()
        return (sorted_probs[t] + sorted_probs[t + 1]) / 2
