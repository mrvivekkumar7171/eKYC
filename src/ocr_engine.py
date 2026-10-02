import logging, easyocr, os
from src.utils import read_yaml


config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifacts['LOG_DIR']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format=logging_str, filemode="a")


def extract_text(image_path, confidence_threshold=0.3, languages=['en']):
    """
    Extracts and filters text from an image using OCR, based on a confidence threshold.

    Parameters:
    - image_path (str): Path to the image file.
    - confidence_threshold (float): Minimum confidence for text inclusion. Default is 0.3.
    - languages (list): OCR languages. Default is ['en'].

    Returns:
    - str: Filtered text separated by '|' if confidence is met, otherwise an empty string.

    Raises:
    - Exception: Outputs error message if OCR processing fails.
    """

    logging.info("Text Extraction Started...")
    # Initialize EasyOCR reader
    reader = easyocr.Reader(languages)
    
    try:
        logging.info("Inside Try-Catch...")
        # Read the image and extract text
        result = reader.readtext(image_path)
        filtered_text = "|"  # Initialize an empty string to store filtered text
        for text in result:
            bounding_box, recognized_text, confidence = text
            if confidence > confidence_threshold:
                filtered_text += recognized_text + "|"  # Append filtered text with newline

        return filtered_text 
    except Exception as e:
        print("An error occurred during text extraction:", e)
        logging.info(f"An error occurred during text extraction: {e}")
        return ""
    finally:
        logging.info("Text Extraction Completed.")