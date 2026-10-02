from src.utils import artifact_path, read_yaml, file_exists
import logging, cv2, os
import numpy as np


config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifact_path(artifacts['LOG_DIR'])

intermediate_dir_path = artifact_path(artifacts['INTERMIDEIATE_DIR'])
conour_file_name = artifacts['CONTOUR_FILE']

parameters = config['parameters']
gaussian_blur_kernel_size = parameters['GAUSSIAN_BLUR_KERNEL_SIZE']
adaptive_threshold_block_size = parameters['ADAPTIVE_THRESHOLD_BLOCK_SIZE']
adaptive_threshold_c = parameters['ADAPTIVE_THRESHOLD_C']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=log_dir / log_file_name, level=logging.INFO, format=logging_str, filemode="a")


def read_image(image_path, is_uploaded=False):
    """
    Reads an image from a file path or a file-like object (from browser uploads).

    Args:
        image_path (str or file-like object): The path to the image file or a file-like object.
        is_uploaded (bool, optional): If True, the image is uploaded as a file-like object. Defaults to False.

    Returns:
        np.ndarray: The image data as a NumPy array, or None if the image cannot be read.
    """
    if is_uploaded:
        try:
            image_bytes = image_path.read()
            img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logging.info("Failed to read image: {}".format(image_path))
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            logging.info(f"Error reading image: {e}")
            print("Error reading image:", e)
            return None
    else:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logging.info("Failed to read image: {}".format(image_path))
                raise Exception("Failed to read image: {}".format(image_path))
            return img
        except Exception as e:
            logging.info(f"Error reading image: {e}")
            print("Error reading image:", e)
            return None


def extract_id_card(img):
    """
    Extracts the ID card from an image containing other backgrounds.

    Args:
        img (np.ndarray): The input image.

    Returns:
        np.ndarray: The cropped image containing the ID card, or None if no ID card is detected.
        str: The filename of the saved image, or None if the image is not saved.
    """

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise reduction
    blur = cv2.GaussianBlur(gray_img, (gaussian_blur_kernel_size, gaussian_blur_kernel_size), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 
        255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 
        adaptive_threshold_block_size, 
        adaptive_threshold_c
    )

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Select the largest contour (assuming the ID card is the largest object)
    largest_contour = None
    largest_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_contour = cnt
            largest_area = area

    # If no large contour is found, assume no ID card is present
    if largest_contour is None:
        return None

    # Get bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    logging.info(f"contours are found at, {(x, y, w, h)}")
    # logging.info("Area largest_area)

    # Apply additional filtering (optional):
    # - Apply bilateral filtering for noise reduction
    # filtered_img = cv2.bilateralFiltering(img[y:y+h, x:x+w], 9, 75, 75)
    # - Morphological operations (e.g., erosion, dilation) for shape refinement
    contour_id = img[y:y+h, x:x+w]

    filename = save_image(contour_id, conour_file_name, intermediate_dir_path)

    return contour_id, filename


def save_image(image, filename, path="."):
    """Save an OpenCV image and return its full path."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    if file_exists(full_path):
          os.remove(full_path)
    if not cv2.imwrite(full_path, image):
          raise IOError(f"Could not save image to {full_path}")
    logging.info(f"Image saved successfully: {full_path}")
    return full_path


def save_uploaded_file(uploaded_file, filename, path=intermediate_dir_path):
  """Persist a Streamlit upload and return its project-local path."""
  if uploaded_file is None:
      return None

  os.makedirs(path, exist_ok=True)
  full_path = os.path.join(path, filename)
  with open(full_path, "wb") as output_file:
      output_file.write(uploaded_file.getvalue())
  logging.info(f"Uploaded file saved successfully: {full_path}")
  return full_path