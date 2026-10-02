from dotenv import load_dotenv
from src.utils import read_yaml
import mysql.connector
import pandas as pd
import os, logging
load_dotenv()


config = read_yaml("config.yaml")
artifacts = config['artifacts']

log_file_name = artifacts['LOG_FILE_NAME']
log_dir = artifacts['LOG_DIR']


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, log_file_name), level=logging.INFO, format=logging_str, filemode="a")


# Establish a connection to MySQL Server
mydb = mysql.connector.connect(
    host=os.getenv("DATABASE_URL"),
    user=os.getenv("USER"),
    password=os.getenv("PASSWORD"),
    database=os.getenv("DATABASE_NAME")
)
mycursor=mydb.cursor()
logging.info("Connection Established with the database")


def insert_records(text_info):
    """Inserts a new record into the 'users' table in the database.

    Args:
        text_info (dict): A dictionary containing the information of the user to insert.
    """
    sql = "INSERT INTO users(id, name, father_name, dob, id_type, embedding) VALUES (%s, %s, %s, %s, %s, %s)"
    value = (text_info['ID'],
        text_info['Name'],
        text_info["Father's Name"],
        text_info['DOB'],  # Make sure this is formatted as a string 'YYYY-MM-DD'
        text_info['ID Type'],
        str(text_info['Embedding'])) # Too check if there is duplicacy before inserting the record
    mycursor.execute(sql, value)
    mydb.commit()


def fetch_records(text_info):
    """Fetches records from the database based on the provided text_info.

    Args:
        text_info (dict): A dictionary containing the information of the user to fetch.

    Returns:
        pd.DataFrame: A DataFrame containing the fetched records.
    """
    sql = "SELECT * FROM users WHERE id =%s"
    value = (text_info['ID'],)
    mycursor.execute(sql, value)
    result = mycursor.fetchall()
    if result:
        return pd.DataFrame(result, columns=[desc[0] for desc in mycursor.description])
    else:
        return pd.DataFrame()