# cli.py
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="WaifuDiffusion Tagger and Folder Organizer CLI")
    parser.add_argument("--folder", type=str, help="Path to the folder containing images or videos.")
    parser.add_argument("--scan", type=str, help="Path to a single image to scan for tags.")
    parser.add_argument("--bytag", type=str, help="Filter tags by specified tag.")
    parser.add_argument("--clean", action="store_true", help="Clean and organize folders based on names in parentheses.")
    parser.add_argument("--exclude", type=str, nargs='*', help="Exclude tags from sorting.")
    parser.add_argument("--characters", action="store_true", help="Only process items with identified character tags.")

    parser.add_argument("--general-thresh", type=float, default=0.35, help="General tags threshold.")
    parser.add_argument("--character-thresh", type=float, default=0.75, help="Character tags threshold.")
    parser.add_argument("--mcut-general", action="store_true", help="Use MCut threshold for general tags.")
    parser.add_argument("--mcut-character", action="store_true", help="Use MCut threshold for character tags.")
    
    return parser.parse_args()
