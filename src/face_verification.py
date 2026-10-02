from src.utils import artifact_path, file_exists, read_yaml
from src.preprocess import save_image
from deepface import DeepFace
import face_recognition
import logging, cv2, os
import numpy as np


config = read_yaml("config.yaml")
artifacts = config['artifacts']
parameters = config['parameters']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifact_path(artifacts['LOG_DIR'])
cascade_path = artifact_path(artifacts['HAARCASCADE_PATH'])

scaleFactor = parameters['SCALE_FACTOR']
minNeighbors = parameters['MIN_NEIGHBORS']
increase_scale_factor = parameters['INCREASED_SCALE_FACTOR']
deepface_model = parameters['DEEPFACE_MODEL']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=log_dir / log_file_name, level=logging.INFO, format=logging_str, filemode="a")

def _detect_and_extract_face(face_cascade, img, option, pan_version=1):
    # Taking only right-bottom side for PAN card due to Mahatama Gandhi is taking as largest picture.
    if option=="PAN" and pan_version==1:
        img = img[img.shape[0]//2:, img.shape[1]//2:]
    else:
        img = img

    # Convert the image to grayscale (Haar cascade works better with grayscale images)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Find the face with the largest area
    max_area = 0
    largest_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)
    return largest_face

def detect_and_extract_face(img, option):
    """ Detect and Extract the largest face from an image.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        str: The path to the saved extracted face image, or None if no face is found.
    """
    # Load the Haar cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path)

    largest_face = _detect_and_extract_face(face_cascade, img, option)
    if largest_face is None and option == "PAN":
        largest_face = _detect_and_extract_face(face_cascade, img, option, pan_version=2)

    # Extract the largest face
    if largest_face is not None:
        (x, y, w, h) = largest_face
        
        # Increase dimensions by X %
        new_w = int(w * increase_scale_factor)
        new_h = int(h * increase_scale_factor)
        
        # Calculate new (x, y) coordinates to keep the center of the face the same
        new_x = max(0, x - int((new_w - w) / 2))
        new_y = max(0, y - int((new_h - h) / 2))

        # Extract the enlarged face
        extracted_face = img[new_y:new_y+new_h, new_x:new_x+new_w]

        return extracted_face
    else:
        return None


def face_comparison(image1_path, image2_path):
    """Compare two images using face_recognition library to verify if they contain the same face.

    Args:
        image1_path (str): Path to the first face image.
        image2_path (str): Path to the second face image.

    Returns:
        bool: True if the faces are verified, False otherwise.
    """

    img1_exists = file_exists(image1_path)
    img2_exists = file_exists(image2_path)

    if not img1_exists or not img2_exists:
        logging.info("Check the path for the images provided")
        return False

    image1 = face_recognition.load_image_file(image1_path)
    image2 = face_recognition.load_image_file(image2_path)

    if image1 is not None and image2 is not None:
        face_encodings1 = face_recognition.face_encodings(image1)
        face_encodings2 = face_recognition.face_encodings(image2)

    else:
        logging.info("Image is not loaded properly")
        return False

    # Check if faces are detected in both images
    if len(face_encodings1) == 0 or len(face_encodings2) == 0:
        logging.info("No faces detected in one or both images.")
        return False
    else:
    # Proceed with comparing faces if faces are detected
        matches = face_recognition.compare_faces(face_encodings1, face_encodings2[0])
    if any(matches):
        return True
    else:
        return False


def get_face_embeddings(image_path):
    """Generate Face Embeddings.

    Args:
        image_path (str): Path to the face image.

    Returns:
        list: A list of face embeddings if exists, otherwise None.
    """

    img_exists = file_exists(image_path)

    if not img_exists:
        logging.info("Check the path for the images provided")
        return None
    
    embedding_objs = DeepFace.represent(img_path=image_path, model_name=deepface_model)
    embedding = embedding_objs[0]["embedding"]

    if len(embedding) > 0:
        return embedding
    return None


if __name__ == "__main__":

    id_card = "data\\docs\\pan_2.jpg"
    face_path = "data\\faces\\extracted_face.jpg"
    id_card = cv2.imread(id_card)
    extracted_face_path = detect_and_extract_face(image_path=id_card, option="PAN")
    face_comparison(extracted_face_path, face_path)