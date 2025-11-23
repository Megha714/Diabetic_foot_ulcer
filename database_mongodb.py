"""
Database Module for DFU Detection System - MongoDB Version
Handles user authentication and patient management using MongoDB Atlas
"""

from pymongo import MongoClient
from datetime import datetime
import hashlib
import secrets
from bson.objectid import ObjectId

# MongoDB Atlas Configuration
MONGO_USER = "1by22cs099_db_user"
MONGO_PW = "lHyJMo6qgN3U1tLQ"
MONGO_HOST = "dfu-cluster.3cbzdtc.mongodb.net"
MONGO_PARAMS = "retryWrites=true&w=majority"
DB_NAME = "dfu_db"

# Build connection URI
from urllib.parse import quote_plus
safe_pw = quote_plus(MONGO_PW)
MONGO_URI = f"mongodb+srv://{MONGO_USER}:{safe_pw}@{MONGO_HOST}/?{MONGO_PARAMS}"

# Global MongoDB client and database
client = None
db = None

def get_db():
    """Get MongoDB database connection"""
    global client, db
    if client is None:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
    return db

def hash_password(password):
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return f"{salt}${pwd_hash}"

def verify_password(password, hashed):
    """Verify password against hash"""
    try:
        salt, pwd_hash = hashed.split('$')
        return hashlib.sha256((password + salt).encode()).hexdigest() == pwd_hash
    except:
        return False

def init_database():
    """Initialize MongoDB collections and create default admin user"""
    database = get_db()
    
    # Create collections if they don't exist (MongoDB creates them automatically on first insert)
    users_col = database["users"]
    patients_col = database["patients"]
    predictions_col = database["predictions"]
    
    # Create indexes for better performance
    users_col.create_index("username", unique=True)
    users_col.create_index("email", unique=True)
    patients_col.create_index("patient_id", unique=True)
    predictions_col.create_index("patient_id")
    predictions_col.create_index("prediction_date")
    
    # Check if admin user exists
    admin_exists = users_col.find_one({"username": "admin"})
    
    if not admin_exists:
        admin_password = hash_password('admin123')
        users_col.insert_one({
            "username": "admin",
            "email": "admin@dfu.com",
            "password_hash": admin_password,
            "full_name": "System Administrator",
            "role": "admin",
            "created_at": datetime.now(),
            "last_login": None
        })
        print("✅ Default admin user created (username: admin, password: admin123)")
    
    print("✅ MongoDB database initialized successfully!")
    print(f"📊 Connected to: {MONGO_HOST}")
    print(f"🗄️  Database: {DB_NAME}")

# User Management Functions
def create_user(username, email, password, full_name, role='patient'):
    """Create a new user"""
    database = get_db()
    users_col = database["users"]
    
    password_hash = hash_password(password)
    result = users_col.insert_one({
        "username": username,
        "email": email,
        "password_hash": password_hash,
        "full_name": full_name,
        "role": role,
        "created_at": datetime.now(),
        "last_login": None
    })
    return str(result.inserted_id)

def authenticate_user(username, password):
    """Authenticate user login"""
    database = get_db()
    users_col = database["users"]
    
    user = users_col.find_one({"$or": [{"username": username}, {"email": username}]})
    
    if user and verify_password(password, user.get('password_hash', '')):
        # Update last login
        users_col.update_one(
            {"_id": user["_id"]},
            {"$set": {"last_login": datetime.now()}}
        )
        # Convert ObjectId to string
        user['id'] = str(user['_id'])
        user.pop('_id')
        user.pop('password_hash')
        return user
    return None

def get_user_by_id(user_id):
    """Get user by ID"""
    database = get_db()
    users_col = database["users"]
    
    user = users_col.find_one({"_id": ObjectId(user_id)})
    if user:
        user['id'] = str(user['_id'])
        user.pop('_id')
        user.pop('password_hash', None)
        return user
    return None

def get_all_users():
    """Get all users"""
    database = get_db()
    users_col = database["users"]
    
    users = list(users_col.find({}, {"password_hash": 0}))
    for user in users:
        user['id'] = str(user['_id'])
        user.pop('_id')
    return users

# Patient Management Functions
def create_patient(patient_data, created_by):
    """Create a new patient"""
    database = get_db()
    patients_col = database["patients"]
    
    patient_doc = {
        "patient_id": patient_data['patient_id'],
        "full_name": patient_data['full_name'],
        "age": patient_data.get('age'),
        "gender": patient_data.get('gender'),
        "phone": patient_data.get('phone'),
        "email": patient_data.get('email'),
        "address": patient_data.get('address'),
        "medical_history": patient_data.get('medical_history'),
        "diabetes_type": patient_data.get('diabetes_type'),
        "diabetes_duration": patient_data.get('diabetes_duration'),
        "blood_sugar_level": patient_data.get('blood_sugar_level'),
        "hba1c_level": patient_data.get('hba1c_level'),
        "has_diabetes": patient_data.get('has_diabetes', 'No'),
        "created_by": ObjectId(created_by),
        "created_at": datetime.now(),
        "updated_at": datetime.now()
    }
    
    result = patients_col.insert_one(patient_doc)
    return str(result.inserted_id)

def get_patient_by_id(patient_id):
    """Get patient by database ID"""
    database = get_db()
    patients_col = database["patients"]
    
    patient = patients_col.find_one({"_id": ObjectId(patient_id)})
    if patient:
        patient['id'] = str(patient['_id'])
        patient['created_by'] = str(patient.get('created_by', ''))
        patient.pop('_id')
        return patient
    return None

def get_patient_by_patient_id(patient_id):
    """Get patient by patient_id (e.g., P001)"""
    database = get_db()
    patients_col = database["patients"]
    
    patient = patients_col.find_one({"patient_id": patient_id})
    if patient:
        patient['id'] = str(patient['_id'])
        patient['created_by'] = str(patient.get('created_by', ''))
        patient.pop('_id')
        return patient
    return None

def get_all_patients():
    """Get all patients"""
    database = get_db()
    patients_col = database["patients"]
    users_col = database["users"]
    
    patients = list(patients_col.find({}))
    for patient in patients:
        patient['id'] = str(patient['_id'])
        created_by_id = patient.get('created_by')
        if created_by_id:
            creator = users_col.find_one({"_id": created_by_id})
            patient['created_by_name'] = creator.get('full_name', 'Unknown') if creator else 'Unknown'
        else:
            patient['created_by_name'] = 'Unknown'
        patient['created_by'] = str(created_by_id) if created_by_id else ''
        patient.pop('_id')
    return patients

def update_patient(patient_id, patient_data):
    """Update patient information"""
    database = get_db()
    patients_col = database["patients"]
    
    update_data = {k: v for k, v in patient_data.items() if k != 'id'}
    update_data['updated_at'] = datetime.now()
    
    result = patients_col.update_one(
        {"_id": ObjectId(patient_id)},
        {"$set": update_data}
    )
    return result.modified_count > 0

def delete_patient(patient_id):
    """Delete a patient"""
    database = get_db()
    patients_col = database["patients"]
    
    result = patients_col.delete_one({"_id": ObjectId(patient_id)})
    return result.deleted_count > 0

def search_patients(query):
    """Search patients by name, ID, or phone"""
    database = get_db()
    patients_col = database["patients"]
    users_col = database["users"]
    
    search_regex = {"$regex": query, "$options": "i"}
    patients = list(patients_col.find({
        "$or": [
            {"full_name": search_regex},
            {"patient_id": search_regex},
            {"phone": search_regex}
        ]
    }))
    
    for patient in patients:
        patient['id'] = str(patient['_id'])
        created_by_id = patient.get('created_by')
        if created_by_id:
            creator = users_col.find_one({"_id": created_by_id})
            patient['created_by_name'] = creator.get('full_name', 'Unknown') if creator else 'Unknown'
        else:
            patient['created_by_name'] = 'Unknown'
        patient['created_by'] = str(created_by_id) if created_by_id else ''
        patient.pop('_id')
    return patients

# Prediction Management Functions
def save_prediction(patient_id, prediction_data):
    """Save a prediction result"""
    database = get_db()
    predictions_col = database["predictions"]
    
    prediction_doc = {
        "patient_id": ObjectId(patient_id),
        "prediction_class": prediction_data.get('prediction_class'),
        "prediction_label": prediction_data.get('prediction_label'),
        "confidence": prediction_data.get('confidence'),
        "image_path": prediction_data.get('image_path'),
        "rejected": prediction_data.get('rejected', False),
        "rejection_reason": prediction_data.get('rejection_reason'),
        "prediction_date": datetime.now()
    }
    
    result = predictions_col.insert_one(prediction_doc)
    return str(result.inserted_id)

def get_patient_predictions(patient_id):
    """Get all predictions for a patient"""
    database = get_db()
    predictions_col = database["predictions"]
    
    predictions = list(predictions_col.find(
        {"patient_id": ObjectId(patient_id)}
    ).sort("prediction_date", -1))
    
    for pred in predictions:
        pred['id'] = str(pred['_id'])
        pred['patient_id'] = str(pred['patient_id'])
        pred.pop('_id')
    return predictions

def get_all_predictions():
    """Get all predictions"""
    database = get_db()
    predictions_col = database["predictions"]
    
    predictions = list(predictions_col.find({}).sort("prediction_date", -1).limit(100))
    for pred in predictions:
        pred['id'] = str(pred['_id'])
        pred['patient_id'] = str(pred['patient_id'])
        pred.pop('_id')
    return predictions

def get_dashboard_stats():
    """Get dashboard statistics"""
    database = get_db()
    patients_col = database["patients"]
    predictions_col = database["predictions"]
    
    total_patients = patients_col.count_documents({})
    total_predictions = predictions_col.count_documents({})
    
    # Count normal and abnormal cases
    normal_cases = predictions_col.count_documents({"prediction_class": 0})
    abnormal_cases = predictions_col.count_documents({"prediction_class": 1})
    
    return {
        'total_patients': total_patients,
        'total_predictions': total_predictions,
        'normal_cases': normal_cases,
        'abnormal_cases': abnormal_cases
    }

def get_recent_predictions(limit=10):
    """Get recent predictions"""
    database = get_db()
    predictions_col = database["predictions"]
    patients_col = database["patients"]
    
    predictions = list(predictions_col.find({}).sort("prediction_date", -1).limit(limit))
    
    for pred in predictions:
        pred['id'] = str(pred['_id'])
        patient_obj_id = pred['patient_id']
        pred['patient_id'] = str(patient_obj_id)
        
        # Get patient info
        patient = patients_col.find_one({"_id": patient_obj_id})
        if patient:
            pred['patient_name'] = patient.get('full_name', 'Unknown')
            pred['patient_patient_id'] = patient.get('patient_id', 'Unknown')
        else:
            pred['patient_name'] = 'Unknown'
            pred['patient_patient_id'] = 'Unknown'
        
        pred.pop('_id')
    
    return predictions
