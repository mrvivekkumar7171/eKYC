from datetime import datetime
from src.utils import read_yaml
import pandas as pd
import json, os, logging, re


config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifacts['LOG_DIR']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format=logging_str, filemode="a")


def extract_information(data_string, option):
    ###### add option parameter to the ID Type in the extracted_info dictionary
    words = [word.strip() for word in data_string.split("|") if word.strip()]
    normalized_words = [re.sub(r"[^a-z0-9]", "", word.lower()) for word in words]

    # Initialize the dictionary to store the extracted information
    extracted_info = {
        "ID": "",
        "Name": "",
        "Father's Name": "",
        "DOB": "",
        "ID Type": "PAN"
    }

    government_index = next(
        (index for index, word in enumerate(normalized_words)
         if word in {"govtofindia", "governmentofindia", "governmentonindia"}),
        None,
    )
    if government_index is not None and government_index + 1 < len(words):
        extracted_info["Name"] = words[government_index + 1]
        if government_index + 2 < len(words):
            possible_father = normalized_words[government_index + 2]
            if possible_father not in {"dob", "male", "female"} and not re.fullmatch(r"\d+", possible_father):
                extracted_info["Father's Name"] = words[government_index + 2]

    for word in words:
        for date_format in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                extracted_info["DOB"] = datetime.strptime(word, date_format).strftime('%Y-%m-%d')
                break
            except ValueError:
                continue
        if extracted_info["DOB"]:
            break

    aadhaar_match = re.search(r"(?<!\d)(\d{4}\s*\d{4}\s*\d{4})(?!\d)", " ".join(words))
    pan_match = re.search(r"[A-Z]{5}\d{4}[A-Z]", " ".join(words).upper())
    if aadhaar_match:
        extracted_info["ID"] = re.sub(r"\s+", "", aadhaar_match.group())
        extracted_info["ID Type"] = "Aadhar"
    elif pan_match:
        extracted_info["ID"] = pan_match.group()
        extracted_info["ID Type"] = "PAN"

    return extracted_info