from src.face_verification import detect_and_extract_face, face_comparison, get_face_embeddings
from src.preprocess import read_image, extract_id_card, save_image, save_uploaded_file
from src.mysqldb_operations import insert_records, fetch_records
from src.postprocess import extract_information
from src.ocr_engine import extract_text
from src.utils import read_yaml
from PIL import Image, ImageOps
from datetime import datetime
from sqlalchemy import text
import streamlit as st
import cv2, logging, os
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


def main_content(id_image_file, face_image_file, conn, option):
    # One placeholder for all status messages
    status = st.empty()

    # Saving the uploaded face and ID card images
    saved_face_path = save_uploaded_file(face_image_file, uploaded_face_img_name, dir_path)
    logging.info("Face image loaded.")

    saved_document_path = save_uploaded_file(id_image_file, uploaded_id_card_name, dir_path)
    logging.info("ID card image loaded.")


    # Loading the saved Document and extracting the ID Card from it and save it
    document_img = read_image(saved_document_path)
    id_card_document = extract_id_card(document_img)
    save_image(id_card_document, contour_file_name, dir_path)
    if id_card_document is None:
        status.error("Could not detect the ID card in the uploaded image.")
        logging.info("Could not detect the ID card in the uploaded image.")
        return
    status.success("ID card extracted successfully.")
    logging.info("ID card extracted successfully.")


    # It Detects, extracts and saves the face from the ID card image and return path.
    extracted_face = detect_and_extract_face(img=id_card_document)
    extracted_face_path = save_image(extracted_face, extracted_face_img_name, dir_path)
    if extracted_face_path is None:
        status.error("No face was detected on the ID card.")
        logging.info("No face was detected on the ID card.")
        return
    status.success("Face extracted from ID card successfully.")
    logging.info("Faces extracted and saved.")


    # Verifying the face from the ID card with the uploaded face image
    is_face_verified = face_comparison(image1_path=extracted_face_path, image2_path=saved_face_path)
    logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")
    if is_face_verified:
        status.success("Face verification successful. Proceeding with information extraction.")


        # Extracting the information and embeddings  from the ID card if the user is verified
        extracted_text = extract_text(id_card_document)
        text_info = extract_information(extracted_text, option)
        text_info['Embedding'] =  get_face_embeddings(extracted_face_path)
        logging.info("Text extracted and information parsed from ID card.")

        # Fetch records in the database
        records = fetch_records(text_info)


        # If the user is already present in the database, display the records, else insert the records in the database
        if records.shape[0] > 0:
            status.success("User already present in the database.")
            logging.info(f"User already present in the database with ID {text_info['ID']}")
        else:
            insert_records(text_info)
            status.success("New user record inserted into the database.")
            logging.info(f"New user record inserted: {text_info['ID']}")
            records = fetch_records(text_info)


        # Display Information, ID card and uploaded face image
        st.header("Extracted Information")
        st.dataframe(records.iloc[:, :-1])

        col1, col2 = st.columns(2)
        with col1:
            preview = prepare_preview(id_card_document)
            st.image(preview, caption="ID card Image", width="stretch")
        with col2:
            face_img = read_image(extracted_face_path)
            preview = prepare_preview(face_img)
            st.image(preview, caption="Uploaded Face Image", width="stretch")
    else:
        status.error("Face verification failed. Please try again.")


def main():
    # Initialize connection.
    conn = st.connection('mysql', type='sql')

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
        main_content(id_card_file, face_file, conn, option)


if __name__ == "__main__":
    main()