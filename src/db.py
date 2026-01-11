# src/db.py
from pymongo import MongoClient
from datetime import datetime
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "mine_alerts")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "quadrant_alerts")

def get_client():
    return MongoClient(MONGO_URI)

def save_alert_df(df: pd.DataFrame, meta: dict = None):
    """
    Save alert DataFrame to MongoDB as a document.
    Returns inserted_id (string).
    """
    client = get_client()
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]
    doc = {
        "created_at": datetime.utcnow(),
        "data": df.to_dict(orient="records")
    }
    if meta:
        doc["meta"] = meta
    res = col.insert_one(doc)
    client.close()
    return str(res.inserted_id)

def get_latest_alert_doc():
    client = get_client()
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]
    doc = col.find_one(sort=[("created_at", -1)])
    client.close()
    return doc  # returns None if no doc

def get_latest_alert_df():
    doc = get_latest_alert_doc()
    if not doc:
        return None, None
    df = pd.DataFrame(doc["data"])
    return df, doc
def save_alert_rows(df: pd.DataFrame):
    """
    Save each row of the DataFrame as a separate document.
    """
    client = get_client()
    db = client[MONGO_DB]
    col = db[MONGO_COLLECTION]
    records = df.to_dict(orient="records")
    if records:
        col.insert_many(records)
    client.close()