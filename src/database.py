"""MongoDB database connection and collections management."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.database import Database
from pymongo.collection import Collection

# Load environment variables
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)

# MongoDB Configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "patient_contact_tracing")

# Global client instance
_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_client() -> MongoClient:
    """Get or create MongoDB client instance."""
    global _client
    if _client is None:
        _client = MongoClient(MONGODB_URI)
    return _client


def get_database() -> Database:
    """Get the database instance."""
    global _db
    if _db is None:
        _db = get_client()[MONGODB_DATABASE]
    return _db


def close_connection():
    """Close MongoDB connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None


# Collection accessors
def get_users_collection() -> Collection:
    """Get users collection for authentication."""
    return get_database()["users"]


def get_persons_collection() -> Collection:
    """Get persons collection (patients, doctors, visitors, workers)."""
    return get_database()["persons"]


def get_face_embeddings_collection() -> Collection:
    """Get face embeddings collection."""
    return get_database()["face_embeddings"]


def get_face_images_collection() -> Collection:
    """Get face images collection for training."""
    return get_database()["face_images"]


def get_mdr_patients_collection() -> Collection:
    """Get MDR patients collection."""
    return get_database()["mdr_patients"]


def get_contacts_collection() -> Collection:
    """Get contacts/contact tracing logs collection."""
    return get_database()["contacts"]


def get_alerts_collection() -> Collection:
    """Get alerts collection for MDR contact alerts."""
    return get_database()["alerts"]


def get_collision_alerts_collection() -> Collection:
    """Get collision alerts collection."""
    return get_database()["collision_alerts"]


def get_unknown_persons_collection() -> Collection:
    """Get unknown persons collection for tracking unregistered individuals."""
    return get_database()["unknown_persons"]


def get_unknown_contacts_collection() -> Collection:
    """Get unknown contacts collection for tracking contacts with unknown persons."""
    return get_database()["unknown_contacts"]


def init_indexes():
    """Initialize database indexes for better performance."""
    try:
        # Users collection indexes
        users = get_users_collection()
        users.create_index("username", unique=True)
        users.create_index("email", unique=True)
        
        # Persons collection indexes
        persons = get_persons_collection()
        persons.create_index("name", unique=True)
        persons.create_index("role")
        persons.create_index("created_at")
        
        # Face embeddings indexes
        face_embeddings = get_face_embeddings_collection()
        face_embeddings.create_index("person_name")
        
        # Face images indexes
        face_images = get_face_images_collection()
        face_images.create_index("person_name")
        face_images.create_index("trained", sparse=True)
        
        # MDR patients indexes
        mdr = get_mdr_patients_collection()
        mdr.create_index("name", unique=True)
        mdr.create_index("marked_at")
        
        # Contacts collection indexes
        contacts = get_contacts_collection()
        contacts.create_index("person")
        contacts.create_index("other_person")
        contacts.create_index([("person", ASCENDING), ("other_person", ASCENDING)])
        contacts.create_index("timestamp")
        
        # Alerts collection indexes
        alerts = get_alerts_collection()
        alerts.create_index("mdr_patient")
        alerts.create_index("contact_name")
        alerts.create_index("timestamp")
        alerts.create_index("is_read")
        
        # Collision alerts indexes
        collision_alerts = get_collision_alerts_collection()
        collision_alerts.create_index("timestamp")
        collision_alerts.create_index([("person1", ASCENDING), ("person2", ASCENDING)])
        
        # Unknown persons indexes
        unknown_persons = get_unknown_persons_collection()
        unknown_persons.create_index("temp_id", unique=True)
        unknown_persons.create_index("first_seen")
        unknown_persons.create_index("last_seen")
        
        # Unknown contacts indexes
        unknown_contacts = get_unknown_contacts_collection()
        unknown_contacts.create_index("unknown_temp_id")
        unknown_contacts.create_index("registered_person")
        unknown_contacts.create_index("timestamp")
        unknown_contacts.create_index([("unknown_temp_id", ASCENDING), ("registered_person", ASCENDING)])
    except Exception as e:
        print(f"[DB] Warning creating indexes (may already exist): {e}")


def init_default_admin():
    """Create default admin user if not exists."""
    users = get_users_collection()
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
    admin_email = os.getenv("ADMIN_EMAIL", "admin@hospital.com")
    
    existing = users.find_one({"username": admin_username})
    if not existing:
        import bcrypt
        password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        users.insert_one({
            "username": admin_username,
            "email": admin_email,
            "password_hash": password_hash,
            "is_admin": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        print(f"[DB] Default admin user '{admin_username}' created.")


def initialize_database():
    """Initialize database with indexes and default data."""
    print(f"[DB] Connecting to MongoDB: {MONGODB_URI}")
    print(f"[DB] Using database: {MONGODB_DATABASE}")
    
    # Test connection
    try:
        get_client().admin.command('ping')
        print("[DB] MongoDB connection successful!")
    except Exception as e:
        print(f"[DB] MongoDB connection failed: {e}")
        raise
    
    # Create indexes
    init_indexes()
    print("[DB] Database indexes created.")
    
    # Create default admin
    init_default_admin()
    
    print("[DB] Database initialization complete.")


__all__ = [
    "get_client",
    "get_database",
    "close_connection",
    "get_users_collection",
    "get_persons_collection",
    "get_face_embeddings_collection",
    "get_face_images_collection",
    "get_mdr_patients_collection",
    "get_contacts_collection",
    "get_alerts_collection",
    "get_collision_alerts_collection",
    "initialize_database",
]
