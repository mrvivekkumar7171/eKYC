from dotenv import load_dotenv
from src.utils import read_yaml
import mysql.connector
import pandas as pd
import os
import logging

load_dotenv()

config = read_yaml("config.yaml")
artifacts = config["artifacts"]
log_dir = artifacts["LOG_DIR"]
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, artifacts["LOG_FILE_NAME"]), level=logging.INFO, format=logging_str, filemode="a")


def _connect():
    return mysql.connector.connect(host=os.getenv("DATABASE_URL"), user=os.getenv("USER"), password=os.getenv("PASSWORD"), database=os.getenv("DATABASE_NAME"))


def _connection(connection):
    return connection or _connect()


def insert_verification_images(document_image, selfie_image, connection=None):
    """Persist original uploads and commit them before processing starts."""
    db = _connection(connection)
    cursor = db.cursor()
    try:
        cursor.execute(
            """INSERT INTO verification_images
               (document_image, selfie_image, verification_status, processing_status)
               VALUES (%s, %s, %s, %s)""",
            (document_image, selfie_image, "RECEIVED", "RECEIVED"),
        )
        db.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        if connection is None:
            db.close()


def _update_image(image_id, values, connection=None):
    if set(values) == {"processing_status"}:
        query = "UPDATE verification_images SET processing_status = %s WHERE image_id = %s"
        params = (values["processing_status"], image_id)
    elif set(values) == {"verification_status", "failure_reason"}:
        query = "UPDATE verification_images SET verification_status = %s, failure_reason = %s WHERE image_id = %s"
        params = (values["verification_status"], values["failure_reason"], image_id)
    else:
        raise ValueError("Unsupported verification image update")
    _execute(query, params, connection)


def update_image_processing_status(image_id, processing_status, connection=None):
    _update_image(image_id, {"processing_status": processing_status}, connection)


def update_image_embedding(image_id, embedding, connection=None):
    _execute(
        "UPDATE verification_images SET embedding = %s WHERE image_id = %s",
        (str(embedding), int(image_id)),
        connection,
    )


def update_image_coordinates(image_id, id_card_bbox=None, face_bbox=None, connection=None):
    if id_card_bbox is not None:
        _execute(
            """UPDATE verification_images
               SET id_card_x = %s, id_card_y = %s, id_card_width = %s, id_card_height = %s
               WHERE image_id = %s""",
            tuple(int(value) for value in id_card_bbox) + (int(image_id),),
            connection,
        )
    if face_bbox is not None:
        _execute(
            """UPDATE verification_images
               SET face_x = %s, face_y = %s, face_width = %s, face_height = %s
               WHERE image_id = %s""",
            tuple(int(value) for value in face_bbox) + (int(image_id),),
            connection,
        )


def update_image_status(image_id, verification_status, failure_reason=None, connection=None):
    processing_status = "COMPLETED" if verification_status in {"SUCCESS", "DUPLICATE_USER"} else "FAILED"
    _execute(
        """UPDATE verification_images
           SET verification_status = %s, processing_status = %s, failure_reason = %s
           WHERE image_id = %s""",
        (verification_status, processing_status, failure_reason, image_id),
        connection,
    )


def _execute(query, params, connection=None):
    db = _connection(connection)
    cursor = db.cursor()
    try:
        cursor.execute(query, params)
        db.commit()
    finally:
        cursor.close()
        if connection is None:
            db.close()


def insert_user(text_info, document_image_id, connection=None):
    db = _connection(connection)
    cursor = db.cursor()
    try:
        cursor.execute(
            """INSERT INTO `user`
                    (name, father_name, dob, id_number, id_type, document_image_id)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                (text_info["Name"], text_info["Father's Name"], text_info["DOB"], text_info["ID"], text_info["ID Type"], document_image_id),
        )
        db.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
        if connection is None:
            db.close()


def fetch_records(text_info, connection=None):
    db = _connection(connection)
    cursor = db.cursor()
    try:
        cursor.execute("SELECT * FROM `user` WHERE id_number = %s AND id_type = %s", (text_info["ID"], text_info["ID Type"]))
        result = cursor.fetchall()
        return pd.DataFrame(result, columns=[desc[0] for desc in cursor.description]) if result else pd.DataFrame()
    finally:
        cursor.close()
        if connection is None:
            db.close()


def user_exists(text_info, connection=None):
    return not fetch_records(text_info, connection).empty


def get_user_image(user_id, image_kind, connection=None):
    if image_kind not in {"document", "selfie"}:
        raise ValueError("image_kind must be 'document' or 'selfie'")
    db = _connection(connection)
    cursor = db.cursor()
    try:
        if image_kind == "document":
            query = """SELECT images.document_image FROM `user` AS users
                       JOIN verification_images AS images ON images.image_id = users.document_image_id
                       WHERE users.id = %s"""
        else:
            query = """SELECT images.selfie_image FROM `user` AS users
                       JOIN verification_images AS images ON images.image_id = users.document_image_id
                       WHERE users.id = %s"""
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        if connection is None:
            db.close()


def get_user_document_image(user_id, connection=None):
    return get_user_image(user_id, "document", connection)


def get_user_selfie_image(user_id, connection=None):
    return get_user_image(user_id, "selfie", connection)


def get_verification_document_image(image_id, connection=None):
    db = _connection(connection)
    cursor = db.cursor()
    try:
        cursor.execute(
            "SELECT document_image FROM verification_images WHERE image_id = %s",
            (image_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()
        if connection is None:
            db.close()


def get_image_coordinates(image_id, connection=None):
    db = _connection(connection)
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""SELECT id_card_x, id_card_y, id_card_width, id_card_height,
                                face_x, face_y, face_width, face_height
                         FROM verification_images WHERE image_id = %s""", (image_id,))
        return cursor.fetchone()
    finally:
        cursor.close()
        if connection is None:
            db.close()


def fetch_failed_attempts(limit=100, connection=None):
    db = _connection(connection)
    cursor = db.cursor()
    try:
        cursor.execute("""SELECT image_id, verification_status, processing_status, failure_reason, created_at
                         FROM verification_images WHERE verification_status <> %s
                         ORDER BY created_at DESC LIMIT %s""", ("SUCCESS", limit))
        return pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
    finally:
        cursor.close()
        if connection is None:
            db.close()


# Retain the old application-facing name while using the new schema.
def insert_records(text_info, document_image_id=None, connection=None):
    return insert_user(text_info, document_image_id, connection)
