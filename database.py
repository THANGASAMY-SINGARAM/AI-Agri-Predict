import hashlib
import hmac
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pymongo import ASCENDING, MongoClient


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_PATH = BASE_DIR / "sample_data.csv"
MONGO_URI = os.getenv("AGRIPREDICT_MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("AGRIPREDICT_DB_NAME", "agripredict")
YIELD_COLLECTION = "crop_yield_records"
USER_COLLECTION = "users"
HASH_ITERATIONS = 120_000


def get_client():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client


def get_database():
    return get_client()[DB_NAME]


def init_database():
    db = get_database()
    db[YIELD_COLLECTION].create_index([("date", ASCENDING), ("region", ASCENDING), ("crop", ASCENDING)])
    db[USER_COLLECTION].create_index("username", unique=True)
    db[USER_COLLECTION].create_index("email", unique=True)
    return db


def import_sample_data(replace=False):
    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {SAMPLE_DATA_PATH.name}")

    db = init_database()
    data = pd.read_csv(SAMPLE_DATA_PATH)
    records = data.to_dict("records")

    if replace:
        db[YIELD_COLLECTION].delete_many({})

    if records:
        db[YIELD_COLLECTION].insert_many(records)

    return count_yield_records()


def load_yield_records():
    db = init_database()
    records = list(db[YIELD_COLLECTION].find({}, {"_id": 0}))
    return pd.DataFrame(records)


def count_yield_records():
    db = init_database()
    return int(db[YIELD_COLLECTION].count_documents({}))


def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS,
    )
    return salt.hex(), password_hash.hex()


def create_user(email, password, full_name=""):
    email = email.strip().lower()
    full_name = full_name.strip()
    if not full_name:
        raise ValueError("Name is required.")
    if not email or "@" not in email:
        raise ValueError("Valid email is required.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")

    db = init_database()
    if db[USER_COLLECTION].find_one({"email": email}):
        raise ValueError("Email already exists.")

    salt, password_hash = _hash_password(password)
    now = datetime.now(timezone.utc)
    db[USER_COLLECTION].insert_one(
        {
            "username": email,
            "email": email,
            "full_name": full_name,
            "password_salt": salt,
            "password_hash": password_hash,
            "created_at": now,
            "last_login_at": None,
        }
    )
    return {"email": email, "username": email, "full_name": full_name}


def authenticate_user(email, password):
    email = email.strip().lower()
    db = init_database()
    user = db[USER_COLLECTION].find_one({"email": email}) or db[USER_COLLECTION].find_one({"username": email})
    if not user:
        return None

    salt = bytes.fromhex(user["password_salt"])
    _, candidate_hash = _hash_password(password, salt=salt)
    if not hmac.compare_digest(candidate_hash, user["password_hash"]):
        return None

    db[USER_COLLECTION].update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login_at": datetime.now(timezone.utc)}},
    )
    return {
        "email": user.get("email", user.get("username", "")),
        "username": user.get("username", user.get("email", "")),
        "full_name": user.get("full_name", ""),
    }
