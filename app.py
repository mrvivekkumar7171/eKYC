from src.face_verification import detect_and_extract_face_with_bbox, face_comparison, get_face_embeddings
from src.preprocess import (
    read_image,
    extract_id_card_with_bbox,
    crop_face_from_document,
    save_image,
    save_uploaded_file,
)
from src.mysqldb_operations import (
    insert_verification_images,
    update_image_coordinates,
    update_image_processing_status,
    update_image_embedding,
    update_image_status,
    insert_user,
    fetch_records,
    get_verification_document_image,
    get_image_coordinates,
)
from src.postprocess import extract_information
from src.ocr_engine import extract_text
from src.utils import read_yaml
from PIL import Image, ImageOps
from datetime import datetime
import streamlit as st
import cv2, logging, os
import numpy as np
from io import BytesIO


st.set_page_config(layout="wide")

config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifacts['LOG_DIR']

extracted_face_img_name = artifacts['FACE_IMG1_NAME']
contour_file_name = artifacts['CONTOUR_FILE']
uploaded_face_img_name = artifacts['FACE_IMG2_NAME']
dir_path = artifacts['INTERMIDEIATE_DIR']
uploaded_id_card_name = artifacts['UPLOADED_ID_CARD_NAME']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format=logging_str, filemode="a")



def sidebar_section():
    option = st.sidebar.selectbox("ID Card Type", ("Aadhar", "PAN"))
    logging.info(f"ID card type selected: {option}")
    return option


def header_section(option):
    if option == "Aadhar":
        st.title("Registration Using Aadhar Card")
        logging.info("Header set for Aadhar Card registration.")
    elif option == "PAN":
        st.title("Registration Using PAN Card")
        logging.info("Header set for PAN Card registration.")


def prepare_preview(image_input, canvas_size=(600, 400)):
    if hasattr(image_input, "getvalue"):
        # Streamlit UploadedFile
        image = Image.open(
            BytesIO(image_input.getvalue())
        ).convert("RGB")

    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")

    else:
        # NumPy/OpenCV image (BGR -> RGB)
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)

    fitted_image = ImageOps.contain(image, canvas_size)

    canvas = Image.new("RGB", canvas_size, "white")

    x = (canvas_size[0] - fitted_image.width) // 2
    y = (canvas_size[1] - fitted_image.height) // 2

    canvas.paste(fitted_image, (x, y))

    return canvas


def main_content(id_image_file, face_image_file, option):
    # One placeholder for all status messages
    status = st.empty()

    image_id = insert_verification_images(id_image_file.getvalue(), face_image_file.getvalue())
    logging.info("Original uploads stored as verification attempt %s.", image_id)

    try:
        update_image_processing_status(image_id, "PROCESSING")
        saved_face_path = save_uploaded_file(face_image_file, uploaded_face_img_name, dir_path)
        saved_document_path = save_uploaded_file(id_image_file, uploaded_id_card_name, dir_path)
        document_img = read_image(saved_document_path)
        id_card_document, id_card_bbox = extract_id_card_with_bbox(document_img)
        if id_card_document is None:
            update_image_status(image_id, "ID_CARD_EXTRACTION_FAILED", "ID card was not detected")
            status.error("Could not detect the ID card in the uploaded image.")
            return
        update_image_coordinates(image_id, id_card_bbox=id_card_bbox)
        save_image(id_card_document, contour_file_name, dir_path)
        status.success("ID card extracted successfully.")

        extracted_face, face_bbox = detect_and_extract_face_with_bbox(id_card_document, option)
        if extracted_face is None:
            update_image_status(image_id, "FACE_EXTRACTION_FAILED", "No face was detected on the ID card")
            status.error("No face was detected on the ID card.")
            return
        document_face_bbox = (
            id_card_bbox[0] + face_bbox[0], id_card_bbox[1] + face_bbox[1], face_bbox[2], face_bbox[3]
        )
        update_image_coordinates(image_id, face_bbox=document_face_bbox)
        extracted_face_path = save_image(extracted_face, extracted_face_img_name, dir_path)
        status.success("Face extracted from ID card successfully.")

        if not face_comparison(image1_path=extracted_face_path, image2_path=saved_face_path):
            update_image_status(image_id, "FACE_VERIFICATION_FAILED", "Uploaded selfie did not match document face")
            status.error("Face verification failed. Please try again.")
            return
        status.success("Face verification successful. Proceeding with information extraction.")

        extracted_text = extract_text(id_card_document)
        text_info = extract_information(extracted_text, option=option)
        if not text_info:
            update_image_status(image_id, "OCR_FAILED", "Could not extract valid identity information")
            status.error("Failed to extract valid information from the ID card.")
            return

        embedding = get_face_embeddings(extracted_face_path)
        if embedding is None:
            update_image_status(image_id, "PROCESSING_ERROR", "Could not generate face embedding")
            status.error("Could not generate a face embedding.")
            return
        update_image_embedding(image_id, embedding)

        records = fetch_records(text_info)
        if records.shape[0] > 0:
            update_image_status(image_id, "DUPLICATE_USER", "Identity is already registered")
            status.success("User already present in the database.")
        else:
            insert_user(text_info, image_id)
            update_image_status(image_id, "SUCCESS")
            status.success("New user record inserted into the database.")
            records = fetch_records(text_info)

        st.header("Extracted Information")
        display_records = records.drop(columns=["embedding"], errors="ignore")
        st.dataframe(display_records)
        original_document_bytes = get_verification_document_image(image_id)
        image_coordinates = get_image_coordinates(image_id)
        original_document = cv2.imdecode(
            np.frombuffer(original_document_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        extracted_face_from_db = crop_face_from_document(
            original_document,
            image_coordinates["face_x"],
            image_coordinates["face_y"],
            image_coordinates["face_width"],
            image_coordinates["face_height"],
        )
        if extracted_face_from_db is None:
            raise ValueError("Could not reconstruct the face from the stored document image")
        col1, col2 = st.columns(2)
        with col1:
            st.image(prepare_preview(id_card_document), caption="ID card Image", width="stretch")
        with col2:
            st.image(prepare_preview(extracted_face_from_db), caption="Extracted Face Image", width="stretch")
    except Exception as error:
        logging.exception("Verification attempt %s failed.", image_id)
        try:
            update_image_status(image_id, "PROCESSING_ERROR", str(error))
        except Exception:
            logging.exception("Could not update failure status for attempt %s.", image_id)
        status.error("Verification could not be completed. The uploaded images were preserved.")


def main():
    option = sidebar_section()
    header_section(option)

    # Two upload sections side-by-side
    col1, col2 = st.columns(2)

    with col1:
        id_card_file = st.file_uploader("Upload ID Card Image", key="id_card", max_upload_size=10)

        # Show immediately after upload
        if id_card_file is not None:
            preview = prepare_preview(id_card_file)
            st.image(preview, width="stretch")

    with col2:
        face_file = st.file_uploader("Upload Face Image", key="face_image", max_upload_size=10)

        # Show immediately after upload
        if face_file is not None:
            preview = prepare_preview(face_file)
            st.image(preview, width="stretch")

    # Operation button
    both_uploaded = id_card_file is not None and face_file is not None
    button_clicked = st.button("Start Verification", type="primary", disabled=not both_uploaded, use_container_width=True)
    if button_clicked:
        main_content(id_card_file, face_file, option)


if __name__ == "__main__":
    main()