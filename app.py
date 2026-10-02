from src.face_verification import detect_and_extract_face, face_comparison, get_face_embeddings
from src.mysqldb_operations import insert_records, fetch_records, check_duplicacy
from src.preprocess import read_image, extract_id_card, save_image, save_uploaded_file
from src.postprocess import extract_information
from src.ocr_engine import extract_text
from src.utils import read_yaml
from datetime import datetime
from sqlalchemy import text
import streamlit as st
import cv2, logging, os


config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifacts['LOG_DIR']

face_img2_name = artifacts['FACE_IMG2_NAME']
intermediate_dir_path = artifacts['INTERMIDEIATE_DIR']
uploaded_id_card_name = artifacts['UPLOADED_ID_CARD_NAME']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format=logging_str, filemode="a")


# Set wider page layout
def wider_page():
    max_width_str = "max-width: 1200px;"
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{ {max_width_str} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Page layout set to wider configuration.")

# Customized Streamlit theme
def set_custom_theme():
    st.markdown(
        """
        <style>
            body {
                background-color: #f0f2f6; /* Set background color */
                color: #333333; /* Set text color */
            }
            .sidebar .sidebar-content {
                background-color: #ffffff; /* Set sidebar background color */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logging.info("Custom theme applied to Streamlit app.")

# Sidebar
def sidebar_section():
    st.sidebar.title("Select ID Card Type")
    option = st.sidebar.selectbox("", ("PAN", "Aadhar"))
    logging.info(f"ID card type selected: {option}")
    return option

# Header
def header_section(option):
    if option == "Aadhar":
        st.title("Registration Using Aadhar Card")
        logging.info("Header set for Aadhar Card registration.")
    elif option == "PAN":
        st.title("Registration Using PAN Card")
        logging.info("Header set for PAN Card registration.")

# Main content
def main_content(document_file, face_image_file, conn):
    if document_file is not None:
        if face_image_file is None:
            st.error("Please upload a face image.")
            return

        saved_face_path = save_uploaded_file(face_image_file, face_img2_name, intermediate_dir_path)
        face_image = read_image(saved_face_path)
        logging.info("Face image loaded.")
        if face_image is not None:
            saved_document_path = save_uploaded_file(document_file, uploaded_id_card_name, intermediate_dir_path)
            document_img = read_image(saved_document_path)
            logging.info("ID card image loaded.")
            extracted_document = extract_id_card(document_img)
            if extracted_document is None:
                st.error("Could not detect the ID card in the uploaded image.")
                return
            document, _ = extracted_document
            logging.info("ID card ROI extracted.")
            # It Detects, extracts and saves the face from the ID card image and return path.
            face_image_path1 = detect_and_extract_face(img=document)
            if face_image_path1 is None:
                st.error("No face was detected on the ID card.")
                return
            face_image_path2 = saved_face_path
            logging.info("Faces extracted and saved.")
            is_face_verified = face_comparison(image1_path=face_image_path1, image2_path=face_image_path2)
            logging.info(f"Face verification status: {'successful' if is_face_verified else 'failed'}.")

            if is_face_verified:
                # Extracting the text only if the user is verified
                extracted_text = extract_text(document)
                text_info = extract_information(extracted_text)
                logging.info("Text extracted and information parsed from ID card.")

                if not isinstance(text_info.get("DOB"), datetime):
                    st.error("Could not extract a valid date of birth from the ID card. Please upload a clearer image.")
                    logging.error("DOB was not extracted from the ID card OCR result.")
                    return

                records = fetch_records(text_info)
                
                if records.shape[0] > 0:
                    st.write(records.iloc[:,:-1])
                
                is_duplicate = check_duplicacy(text_info)
                if is_duplicate:
                    st.write(f"User already present with ID {text_info['ID']}")
                else:
                    st.write(text_info)
                    text_info['DOB'] = text_info['DOB'].strftime('%Y-%m-%d')
                    text_info['Embedding'] =  get_face_embeddings(face_image_path1)
                    insert_records(text_info)
                    logging.info(f"New user record inserted: {text_info['ID']}")

                    col1, col2 = st.columns(2)

                    # Display ID card image
                    with col1:
                        st.header("ID Card Image")
                        document = cv2.cvtColor(document, cv2.COLOR_BGR2RGB)
                        st.image(document, caption="ID card")

                    # Display uploaded face image
                    with col2:
                        st.header("Uploaded Face Image")
                        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                        st.image(face_image, caption="Uploaded Face")

                    # Display extracted information
                    st.header("Extracted Information")
                    st.dataframe(records)
            else:
                st.error("Face verification failed. Please try again.")

        else:
            st.error("Face image not uploaded. Please upload a face image.")
            logging.error("No face image uploaded.")

    else:
        st.warning("Please upload an ID card image.")
        logging.warning("No ID card image uploaded.")

# Main function setup as previously provided...
def main():
    # Initialize connection.
    conn = st.connection('mysql', type='sql')
    wider_page()
    set_custom_theme()
    option = sidebar_section()
    header_section(option)
    image_file = st.file_uploader("Upload ID Card")
    if image_file is not None:
        face_image_file = st.file_uploader("Upload Face Image")
        main_content(image_file, face_image_file, conn)

if __name__ == "__main__":
    main()