import database
from bson.objectid import ObjectId

# Test creating a patient
try:
    db = database.get_db()
    
    patient_data = {
        'patient_id': 'TEST-123',
        'full_name': 'Test Patient',
        'age': '30',
        'gender': 'Male',
        'phone': '1234567890',
        'email': 'test@test.com',
        'address': 'Test Address',
        'medical_history': 'None',
        'diabetes_type': 'Type 2',
        'diabetes_duration': '5',
        'blood_sugar_level': '120',
        'hba1c_level': '6.5',
        'has_diabetes': 'Yes'
    }
    
    # Try to create with a test user ID
    users_col = db["users"]
    user = users_col.find_one({})
    
    if user:
        user_id = str(user['_id'])
        print(f"Using user ID: {user_id}")
        
        result = database.create_patient(patient_data, user_id)
        print(f"SUCCESS! Patient created with ID: {result}")
    else:
        print("No users found in database")
        
except Exception as e:
    print(f"ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
