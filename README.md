### eKYC (Electronic Know Your Customer)
User uploads an image of their ID card and a selfie. The system extracts the text from the ID card and compares the face in the ID card with the selfie to verify the identity of the user. If the user exists in the database, the system will return the user's details. If not, the system will add the user to the database. The application is built using Streamlit.

![alt text](data/raw_data/image.png)

1. **Image Preprocessing**:
    - **GrayScale Conversion**: to remove color information and reduce the complexity of the image.
    - **Smoothing/Blurring**: using `Gaussian Bluring` to reduce noise and improve the accuracy of contour detection.
    - **Thresholding**: using `Adaptive Thresholding` to convert the image to a binary image and enhance the contrast between the ID card and the background.
    - **Contour Detection**: using `cv2.findContours()` to remove background and extract the ID card from the image.
2. **OCR Engine** using `EasyOCR` to extract text from the ID card image. Which use CRNN + CTC (CNN + RNN + Connectionist Temporal Classification).
3. **Extract Face**: Identify and extract largest faces from the image. using OpenCV with `Haar Cascade`.
4. **Face Comparison**: Compare the extracted face with the customer's photo to verify the identity.
5. **Post Processing**: Extracting texts like name, id, Father's name, etc. from the OCR Output and save to the database.


## Features to be added:
Duplicacy Check:
- No duplicate ID cards.
- Face Embedding already exists or not.
Liveness Check:
- Using Motion Detection to check if the user is live or not.
- Eye Blink Detection to check if the user is live or not.

We must containize using `Docker` each part seperately to make it more modular and save computation time and resouces as some of the parts take heavy time and computation, while other parts are light and fast.

---

### 1. Create environment & Upgrade pip tools

```bash
conda create -n ekyc python=3.11.16
conda activate ekyc

python -m pip install --upgrade pip wheel
pip install "setuptools<81"
```

### 2. Install NumPy & OpenCV & dlib for Windows & face recognition

```bash
pip install numpy==2.4.6 dlib-bin==20.0.1 face_recognition_models==0.3.0 easyocr==1.7.2 mysql-connector-python==26.7.0 pandas==3.0.6 pillow==12.3.0 SQLAlchemy==2.1.1 streamlit==1.64.0 deepface==0.0.101 tf_keras==2.21.0
pip install face_recognition==1.3.0 --no-deps
```

### 3. Install remaining packages & Verify

```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless==4.10.0.84
```
> 1. Do not install `opencv-python` together with `opencv-python-headless`.
> 2. Do not install the normal `dlib` package.
> 3. `face_recognition` is installed with `--no-deps` because it tries to install/build normal `dlib`.

### 4. Created a database and table in MySQL
```sql
CREATE DATABASE ekyc;

USE ekyc;

CREATE TABLE users (
    id VARCHAR(255) NOT NULL PRIMARY KEY,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    name VARCHAR(255),
    father_name VARCHAR(255),
    dob DATETIME,
    id_type VARCHAR(255) NOT NULL,
    embedding BLOB
);

SHOW TABLES;
```

### 5. Run the Streamlit App
> Make sure SQL server is running before running the streamlit app
```bash
streamlit run app.py
```