from scripts.config import logger
from scripts.predictor import Predictor
from scripts.processors import process_image, process_folder
from scripts.utils import clean_folders
from scripts.cli import parse_args
import scripts.sort as sort

def main():
    args = parse_args()
    
    if not args.folder and not args.scan:
        logger.error("Please provide either --folder or --scan argument.")
        return

    # Check if the --sort argument is present
    if args.sort:
        if args.folder:
            logger.info("Running media file sorting...")
            sort.sort_media_files(args.folder)
        else:
            logger.error("--sort option requires --folder argument.")
            return
    
    if args.clean:
        if args.folder:
            logger.info("Running folder cleanup...")
            clean_folders(args.folder)
        else:
            logger.error("--clean option requires --folder argument.")
            return

    if args.scan or (args.folder and not args.sort and not args.clean):
        predictor = Predictor()
        if args.scan:
            process_image(predictor, args.scan, args)
        else:
            process_folder(predictor, args.folder, args)

if __name__ == "__main__":
    main()