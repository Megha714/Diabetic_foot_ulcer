"""
Database Module for DFU Detection System
Handles user authentication and patient management
"""

import sqlite3
from datetime import datetime
import hashlib
import secrets
from contextlib import contextmanager

DATABASE_NAME = 'dfu_system.db'

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

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
    """Initialize database with required tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT DEFAULT 'doctor',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                phone TEXT,
                email TEXT,
                address TEXT,
                medical_history TEXT,
                diabetes_type TEXT,
                diabetes_duration INTEGER,
                blood_sugar_level REAL,
                hba1c_level REAL,
                has_diabetes TEXT DEFAULT 'No',
                created_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by) REFERENCES users(id)
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                image_path TEXT,
                is_valid_foot BOOLEAN,
                validation_confidence REAL,
                predicted_class INTEGER,
                class_name TEXT,
                confidence REAL,
                normal_prob REAL,
                abnormal_prob REAL,
                rejection_reason TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create default admin user if not exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        if cursor.fetchone()[0] == 0:
            admin_password = hash_password('admin123')
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            ''', ('admin', 'admin@dfu.com', admin_password, 'System Administrator', 'admin'))
            print("✅ Default admin user created (username: admin, password: admin123)")
        
        conn.commit()
        print("✅ Database initialized successfully!")

# User Management Functions
def create_user(username, email, password, full_name, role='doctor'):
    """Create a new user"""
    with get_db() as conn:
        cursor = conn.cursor()
        password_hash = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, full_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, email, password_hash, full_name, role))
        return cursor.lastrowid

def authenticate_user(username, password):
    """Authenticate user login"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, username))
        user = cursor.fetchone()
        
        if user and verify_password(password, user['password_hash']):
            # Update last login
            cursor.execute('UPDATE users SET last_login = ? WHERE id = ?', 
                         (datetime.now(), user['id']))
            conn.commit()
            return dict(user)
        return None

def get_user_by_id(user_id):
    """Get user by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        return dict(user) if user else None

def get_all_users():
    """Get all users"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT id, username, email, full_name, role, created_at, last_login FROM users')
        return [dict(row) for row in cursor.fetchall()]

# Patient Management Functions
def create_patient(patient_data, created_by):
    """Create a new patient"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (
                patient_id, full_name, age, gender, phone, email, 
                address, medical_history, diabetes_type, diabetes_duration,
                blood_sugar_level, hba1c_level, has_diabetes, created_by
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_data['patient_id'],
            patient_data['full_name'],
            patient_data.get('age'),
            patient_data.get('gender'),
            patient_data.get('phone'),
            patient_data.get('email'),
            patient_data.get('address'),
            patient_data.get('medical_history'),
            patient_data.get('diabetes_type'),
            patient_data.get('diabetes_duration'),
            patient_data.get('blood_sugar_level'),
            patient_data.get('hba1c_level'),
            patient_data.get('has_diabetes', 'No'),
            created_by
        ))
        return cursor.lastrowid

def get_patient_by_id(patient_id):
    """Get patient by ID"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient = cursor.fetchone()
        return dict(patient) if patient else None

def get_patient_by_patient_id(patient_id):
    """Get patient by patient_id"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
        patient = cursor.fetchone()
        return dict(patient) if patient else None

def get_all_patients():
    """Get all patients"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, u.full_name as created_by_name 
            FROM patients p 
            LEFT JOIN users u ON p.created_by = u.id 
            ORDER BY p.created_at DESC
        ''')
        return [dict(row) for row in cursor.fetchall()]

def update_patient(patient_id, patient_data):
    """Update patient information"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE patients 
            SET full_name = ?, age = ?, gender = ?, phone = ?, email = ?, 
                address = ?, medical_history = ?, diabetes_type = ?, 
                diabetes_duration = ?, updated_at = ?
            WHERE id = ?
        ''', (
            patient_data['full_name'],
            patient_data.get('age'),
            patient_data.get('gender'),
            patient_data.get('phone'),
            patient_data.get('email'),
            patient_data.get('address'),
            patient_data.get('medical_history'),
            patient_data.get('diabetes_type'),
            patient_data.get('diabetes_duration'),
            datetime.now(),
            patient_id
        ))
        return cursor.rowcount > 0

def delete_patient(patient_id):
    """Delete a patient"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
        return cursor.rowcount > 0

# Prediction Management Functions
def save_prediction(prediction_data, patient_id, user_id):
    """Save a DFU prediction result"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (
                patient_id, user_id, image_path, is_valid_foot, validation_confidence,
                predicted_class, class_name, confidence, normal_prob, abnormal_prob,
                rejection_reason, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            user_id,
            prediction_data.get('image_path'),
            prediction_data.get('is_valid_foot'),
            prediction_data.get('validation_confidence'),
            prediction_data.get('predicted_class'),
            prediction_data.get('class_name'),
            prediction_data.get('confidence'),
            prediction_data.get('probabilities', {}).get('Normal'),
            prediction_data.get('probabilities', {}).get('Abnormal'),
            prediction_data.get('rejection_reason'),
            prediction_data.get('notes')
        ))
        return cursor.lastrowid

def get_predictions_by_patient(patient_id):
    """Get all predictions for a patient"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.*, u.full_name as doctor_name 
            FROM predictions p 
            LEFT JOIN users u ON p.user_id = u.id 
            WHERE p.patient_id = ? 
            ORDER BY p.created_at DESC
        ''', (patient_id,))
        return [dict(row) for row in cursor.fetchall()]

def get_all_predictions():
    """Get all predictions"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pred.*, pat.full_name as patient_name, u.full_name as doctor_name 
            FROM predictions pred 
            LEFT JOIN patients pat ON pred.patient_id = pat.id 
            LEFT JOIN users u ON pred.user_id = u.id 
            ORDER BY pred.created_at DESC
            LIMIT 100
        ''')
        return [dict(row) for row in cursor.fetchall()]

def get_dashboard_stats():
    """Get statistics for dashboard"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Total patients
        cursor.execute('SELECT COUNT(*) as count FROM patients')
        total_patients = cursor.fetchone()['count']
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) as count FROM predictions')
        total_predictions = cursor.fetchone()['count']
        
        # Abnormal cases
        cursor.execute('SELECT COUNT(*) as count FROM predictions WHERE predicted_class = 1')
        abnormal_cases = cursor.fetchone()['count']
        
        # Recent predictions
        cursor.execute('''
            SELECT pred.*, pat.full_name as patient_name 
            FROM predictions pred 
            LEFT JOIN patients pat ON pred.patient_id = pat.id 
            ORDER BY pred.created_at DESC 
            LIMIT 5
        ''')
        recent_predictions = [dict(row) for row in cursor.fetchall()]
        
        return {
            'total_patients': total_patients,
            'total_predictions': total_predictions,
            'abnormal_cases': abnormal_cases,
            'normal_cases': total_predictions - abnormal_cases,
            'recent_predictions': recent_predictions
        }

if __name__ == '__main__':
    print("Initializing DFU Detection System Database...")
    init_database()
    print("\nDatabase setup complete!")
