from src.utils import read_yaml
from datetime import datetime
import json, os, logging, re
import pandas as pd


config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifacts['LOG_DIR']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format=logging_str, filemode="a")


_NAME_STOPWORDS = {
    "govt", "of", "india", "government", "department", "income", "tax",
    "dob", "date", "birth", "male", "female", "signature", "permanent",
    "account", "number", "aadhaar", "aadhar", "uidai", "vid", "name",
    "father", "fathersname", "fathername", "address"
}


def _canonical_option(option):
    """Return the canonical ID type used by the database."""
    if not isinstance(option, str):
        return None

    value = re.sub(r"[^a-z]", "", option.lower())

    if value in {"aadhar", "aadhaar"}:
        return "Aadhar"
    if value == "pan":
        return "PAN"

    return None


def _normalize_token(value):
    """Normalize whitespace without destroying the original OCR text."""
    return re.sub(r"\s+", " ", str(value)).strip()


def _normalized(value):
    """Lower-case and keep only letters/numbers for OCR comparisons."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _is_government_of_india(value):
    """Handle common OCR variants such as Governmentonindia."""
    value = _normalized(value)

    if value in {
        "govtofindia",
        "governmentofindia",
        "governmentonindia",
    }:
        return True

    # More tolerant fallback for OCR variations.
    return (
        (value.startswith("govt") or value.startswith("government"))
        and value.endswith("india")
    )


def _is_name_candidate(value):
    """Return True only for a reasonably clean human-name-like OCR token."""
    value = _normalize_token(value)

    if not value or len(value) < 2:
        return False

    # Reject tokens containing digits or OCR garbage.
    if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]*", value):
        return False

    normalized = _normalized(value)
    if normalized in _NAME_STOPWORDS:
        return False

    # At least one meaningful alphabetic word.
    return any(
        len(re.sub(r"[^A-Za-z]", "", part)) >= 2
        for part in value.split()
    )


def _extract_dob(words):
    """
    Find and validate a DOB. Invalid dates are ignored.

    Supported formats:
        DD/MM/YYYY
        DD-MM-YYYY
        DD.MM.YYYY
        YYYY-MM-DD
    """
    patterns = (
        (r"(?<!\d)(\d{2}/\d{2}/\d{4})(?!\d)", "%d/%m/%Y"),
        (r"(?<!\d)(\d{2}-\d{2}-\d{4})(?!\d)", "%d-%m-%Y"),
        (r"(?<!\d)(\d{2}\.\d{2}\.\d{4})(?!\d)", "%d.%m.%Y"),
        (r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)", "%Y-%m-%d"),
    )

    for index, word in enumerate(words):
        for pattern, date_format in patterns:
            match = re.search(pattern, word)
            if not match:
                continue

            try:
                parsed = datetime.strptime(match.group(1), date_format)
                return parsed.strftime("%Y-%m-%d"), index
            except ValueError:
                # Example: 1995-17-06 -> invalid month, so do not accept it.
                continue

    return "", None


def _normalize_pan_candidate(value):
    """
    Normalize one 10-character PAN OCR candidate.

    Contextual OCR corrections are applied only at the positions where
    the PAN format requires digits or letters.
    """
    compact = re.sub(r"[^A-Za-z0-9]", "", value).upper()

    if len(compact) != 10:
        return None

    # OCR confusions commonly seen in the 4 numeric PAN positions.
    digit_map = {
        "O": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
        "T": "7",
        "Q": "0",
    }

    # OCR confusions commonly seen in the letter positions.
    letter_map = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
    }

    chars = list(compact)

    for i in range(10):
        if 5 <= i <= 8:  # PAN positions 6-9 are digits
            chars[i] = digit_map.get(chars[i], chars[i])
        else:  # PAN positions 1-5 and 10 are letters
            chars[i] = letter_map.get(chars[i], chars[i])

    pan = "".join(chars)

    if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", pan):
        return pan

    return None


def _extract_pan(words):
    """Extract a PAN from OCR words and tolerate common OCR substitutions."""
    for word in words:
        candidate = _normalize_pan_candidate(word)
        if candidate:
            return candidate

    # Handle a PAN accidentally split into adjacent OCR tokens, e.g.
    # "CRXPP" + "7767D".
    for i in range(len(words) - 1):
        combined = f"{words[i]}{words[i + 1]}"
        candidate = _normalize_pan_candidate(combined)
        if candidate:
            return candidate

    return None


def _extract_aadhaar(words):
    """
    Extract a 12-digit Aadhaar number.

    Supported:
        123456789012
        1234 5678 9012
        1234|5678|9012
    """
    # One OCR field containing the whole number.
    for word in words:
        match = re.search(r"(?<!\d)(\d{12})(?!\d)", word)
        if match:
            return match.group(1)

        match = re.search(
            r"(?<!\d)(\d{4})[\s-]+(\d{4})[\s-]+(\d{4})(?!\d)",
            word,
        )
        if match:
            return "".join(match.groups())

    # Number split across three adjacent OCR fields.
    for i in range(len(words) - 2):
        parts = [words[i].strip(), words[i + 1].strip(), words[i + 2].strip()]

        if all(re.fullmatch(r"\d{4}", part) for part in parts):
            return "".join(parts)

    return None


def _is_masked_aadhaar(words):
    """Detect masked Aadhaar such as XXXX XXXX XXXX."""
    for word in words:
        compact = re.sub(r"[^A-Za-z]", "", word).upper()

        if len(compact) >= 8 and re.fullmatch(r"X+", compact):
            return True

    return False


def _extract_name_and_father(words, government_index, dob_index, option):
    """
    Extract name/father name without assuming the immediate next OCR token
    is always the name.
    """
    # 1. Explicit "Name:" label.
    for i, word in enumerate(words):
        if _normalized(word) == "name" and i + 1 < len(words):
            if _is_name_candidate(words[i + 1]):
                name = words[i + 1]
                father = ""
                break
    else:
        name = ""
        father = ""

    # 2. Otherwise, use words after "Government of India" and before DOB.
    if not name and government_index is not None:
        start = government_index + 1
        end = dob_index if dob_index is not None else len(words)

        candidates = [
            word
            for word in words[start:end]
            if _is_name_candidate(word)
        ]

        if candidates:
            name = candidates[0]

            # For PAN, the second clean name-like token is normally the
            # father's name. For Aadhaar, father's name is optional.
            if len(candidates) > 1:
                father = candidates[1]

    # 3. Look for an explicit father-name label.
    father_label_names = {
        "father",
        "fathersname",
        "fathername",
    }

    for i, word in enumerate(words):
        if _normalized(word) in father_label_names:
            for j in range(i + 1, min(i + 4, len(words))):
                if _is_name_candidate(words[j]):
                    father = words[j]
                    break

            if father:
                break

    return name, father


def extract_information(data_string, option):
    """
    Extract information from OCR text and validate it for Aadhaar/PAN.

    Returns:
        dict: Valid extracted information.
        False: If a required field is missing/invalid or a masked Aadhaar
               document is detected.
    """

    if not isinstance(data_string, str) or not data_string.strip():
        logging.warning("OCR data is empty.")
        return False

    id_type = _canonical_option(option)
    if not id_type:
        logging.warning("Unsupported ID type: %r", option)
        return False

    # Keep the original OCR words, but remove empty pipe-separated fields.
    words = [
        _normalize_token(word)
        for word in data_string.split("|")
        if _normalize_token(word)
    ]

    normalized_words = [_normalized(word) for word in words]

    extracted_info = {
        "ID": "",
        "Name": "",
        "Father's Name": "",
        "DOB": "",
        "ID Type": id_type,
    }

    # Locate Government of India using tolerant OCR matching.
    government_index = next(
        (
            index
            for index, word in enumerate(words)
            if _is_government_of_india(word)
        ),
        None,
    )

    # DOB is required for the database update.
    dob, dob_index = _extract_dob(words)
    extracted_info["DOB"] = dob

    if not dob:
        logging.warning("Valid DOB not found: %s", data_string)
        return False

    # Name extraction is shared between Aadhaar and PAN.
    name, father_name = _extract_name_and_father(
        words, government_index, dob_index, id_type
    )

    extracted_info["Name"] = name
    extracted_info["Father's Name"] = father_name

    if not extracted_info["Name"]:
        logging.warning("Name not found: %s", data_string)
        return False

    if id_type == "PAN":
        pan = _extract_pan(words)

        if not pan:
            logging.warning("Valid PAN not found: %s", data_string)
            return False

        extracted_info["ID"] = pan

        # PAN cards are expected to provide father's name in your database
        # schema, so treat a missing father name as extraction failure.
        if not extracted_info["Father's Name"]:
            logging.warning("Father's Name not found for PAN: %s", data_string)
            return False

    else:  # Aadhar
        # XXXX XXXX XXXX explicitly means masked document -> reject it.
        if _is_masked_aadhaar(words):
            logging.warning("Masked Aadhaar detected: %s", data_string)
            return False

        aadhaar = _extract_aadhaar(words)

        if not aadhaar:
            logging.warning("Valid 12-digit Aadhaar not found: %s", data_string)
            return False

        extracted_info["ID"] = aadhaar

    logging.info("Extracted Information: %s", extracted_info)
    return extracted_info