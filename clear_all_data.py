"""Script to clear all data from MongoDB Atlas while keeping collections intact."""
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env", override=True)

from database import (
    get_database,
    get_users_collection,
    get_persons_collection,
    get_person_id_counters_collection,
    get_face_embeddings_collection,
    get_face_images_collection,
    get_mdr_patients_collection,
    get_contacts_collection,
    get_alerts_collection,
    get_collision_alerts_collection,
    get_unknown_persons_collection,
    get_unknown_contacts_collection,
    get_person_risk_scores_collection,
    get_pathogens_collection,
    initialize_database,
)


def clear_all_data():
    """Clear all data from all collections."""
    print("=" * 60)
    print("MONGODB DATA CLEANUP SCRIPT")
    print("=" * 60)
    print()
    
    # Collections to clear (with display names)
    collections_to_clear = [
        ("users", get_users_collection()),
        ("persons", get_persons_collection()),
        ("person_id_counters", get_person_id_counters_collection()),
        ("face_embeddings", get_face_embeddings_collection()),
        ("face_images", get_face_images_collection()),
        ("mdr_patients", get_mdr_patients_collection()),
        ("contacts", get_contacts_collection()),
        ("alerts", get_alerts_collection()),
        ("collision_alerts", get_collision_alerts_collection()),
        ("unknown_persons", get_unknown_persons_collection()),
        ("unknown_contacts", get_unknown_contacts_collection()),
        ("person_risk_scores", get_person_risk_scores_collection()),
        ("pathogens", get_pathogens_collection()),
    ]
    
    total_deleted = 0
    
    for name, collection in collections_to_clear:
        try:
            count_before = collection.count_documents({})
            result = collection.delete_many({})
            deleted = result.deleted_count
            total_deleted += deleted
            print(f"  ✓ {name}: Deleted {deleted} documents")
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")
    
    print()
    print(f"Total documents deleted: {total_deleted}")
    print()
    print("-" * 60)
    print("Reinitializing database with defaults...")
    print("-" * 60)
    
    # Reinitialize to create default admin and pathogens
    initialize_database()
    
    print()
    print("=" * 60)
    print("DATA CLEANUP COMPLETE!")
    print("=" * 60)
    print()
    print("Default admin user has been recreated:")
    print(f"  Username: {os.getenv('ADMIN_USERNAME', 'admin')}")
    print(f"  Password: {os.getenv('ADMIN_PASSWORD', 'admin123')}")
    print()


if __name__ == "__main__":
    confirm = input("This will DELETE ALL DATA from MongoDB Atlas. Are you sure? (yes/no): ")
    if confirm.lower() == "yes":
        clear_all_data()
    else:
        print("Aborted.")
