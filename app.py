"""
Flask Web Application for DFU Detection System
Complete medical application with user authentication and patient management
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
import io
import base64
import os
from datetime import datetime
from werkzeug.utils import secure_filename

import database as db
from auth import login_required, admin_required, validate_email, validate_password, validate_username, generate_patient_id
from pdf_generator import generate_patient_report

# Lazy imports for heavy libraries (only loaded when needed for predictions)
model_module = None
foot_detection_module = None
torch = None
F = None
transforms = None
Image = None
np = None
ndimage = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-change-this-in-production'  # Change in production!
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model
model = None
device = None
transform = None
model_loaded = False

# Class names
CLASS_NAMES = {
    0: 'Normal (Healthy Skin)',
    1: 'Abnormal (Ulcer)'
}

def lazy_load_ml_modules():
    """Lazy load heavy ML modules only when needed for predictions"""
    global model_module, foot_detection_module, torch, F, transforms, Image, np, ndimage
    
    if torch is None:
        print("Loading ML modules...")
        import torch as torch_module
        import torch.nn.functional as F_module
        from torchvision import transforms as transforms_module
        from PIL import Image as Image_module
        import numpy as np_module
        from scipy import ndimage as ndimage_module
        from model import HybridDFUModel
        from foot_detection_algorithm import analyze_foot_image
        
        torch = torch_module
        F = F_module
        transforms = transforms_module
        Image = Image_module
        np = np_module
        ndimage = ndimage_module
        model_module = HybridDFUModel
        foot_detection_module = analyze_foot_image
        print("ML modules loaded successfully!")
    
    return model_module, foot_detection_module

def load_model():
    """Load the trained model - lazy loading"""
    global model, device, transform, model_loaded
    
    if model_loaded:
        return
    
    # First lazy load the ML modules
    HybridDFUModel, _ = lazy_load_ml_modules()
    
    if model_loaded:
        return
    
    print("Loading AI model...")
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load model with strict rejection threshold to filter non-foot images
    model = HybridDFUModel(num_classes=2, pretrained=False, rejection_threshold=0.75)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
        print(f"Best Accuracy: {checkpoint['best_val_acc']:.4f}")
        print(f"Epoch: {checkpoint['epoch']}")
    else:
        print("Checkpoint not found! Using untrained model.")
    
    model.to(device)
    model.eval()
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model_loaded = True
    print("Model loaded and ready!")

def predict_image(image_file):
    """
    Predict DFU from uploaded image with enhanced validation
    
    Returns:
        dict with prediction results
    """
    # Lazy load ML modules on first prediction
    _, analyze_foot_image = lazy_load_ml_modules()
    
    # Load model on first use
    if not model_loaded:
        load_model()
    
    try:
        # Read and process image
        image = Image.open(image_file).convert('RGB')
        original_size = image.size
        
        # Convert to numpy for computer vision analysis
        img_array = np.array(image)
        
        # Use custom computer vision algorithm to detect if this is a foot
        cv_analysis = analyze_foot_image(img_array)
        
        # If CV confidence is very low (< 35%), definitely reject
        # If CV confidence is borderline (35-50%), check model too before rejecting
        if cv_analysis['confidence'] < 0.35:
            # Very low CV confidence - definitely not a foot
            return {
                'success': True,
                'is_valid_foot': False,
                'validation_confidence': cv_analysis['confidence'],
                'rejected': True,
                'rejection_reason': cv_analysis['rejection_reason'],
                'predicted_class': -1,
                'class_name': 'Invalid Image',
                'confidence': 0.0,
                'probabilities': {
                    'Normal': 0.0,
                    'Abnormal': 0.0
                },
                'original_size': original_size,
                'cv_analysis': cv_analysis['details']
            }
        
        # For borderline CV scores (35-50%), get model opinion before deciding
        # Transform image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict with model
        with torch.no_grad():
            output = model.predict(image_tensor)
        
        # Extract results
        is_valid = output['is_valid_foot'][0].item()
        validation_confidence = output['validation_confidence'][0].item()
        predicted_class = output['predicted_class'][0].item()
        class_probs = output['class_probabilities'][0].cpu().numpy()
        confidence = output['confidence'][0].item()
        rejected = output['rejected'][0].item()
        
        # Smart hybrid decision: combine CV and model results
        prob_diff = abs(class_probs[0] - class_probs[1])
        
        enhanced_rejected = rejected
        rejection_reason = None
        
        # If CV is borderline (35-50%) AND model is also uncertain, reject
        if 0.35 <= cv_analysis['confidence'] < 0.50:
            if confidence < 0.60 or validation_confidence < 0.50:
                enhanced_rejected = True
                rejection_reason = cv_analysis['rejection_reason'] or "Low confidence - may not be a foot image"
        
        # If CV confidence is decent (≥50%) but model is very uncertain, still reject
        elif confidence < 0.25:
            enhanced_rejected = True
            rejection_reason = "Classification confidence extremely low"
        
        # Prepare response
        result = {
            'success': True,
            'is_valid_foot': bool(is_valid and not enhanced_rejected),
            'validation_confidence': float(validation_confidence),
            'rejected': bool(enhanced_rejected),
            'rejection_reason': rejection_reason if enhanced_rejected else None,
            'predicted_class': int(predicted_class) if not enhanced_rejected else -1,
            'class_name': CLASS_NAMES[predicted_class] if not enhanced_rejected else 'Invalid Image',
            'confidence': float(confidence),
            'probabilities': {
                'Normal': float(class_probs[0]),
                'Abnormal': float(class_probs[1])
            },
            'original_size': original_size,
            'cv_analysis': cv_analysis
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Home page - redirect to login or dashboard"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db.authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session['role'] = user['role']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Signup page"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        full_name = request.form.get('full_name')
        role = request.form.get('role', 'doctor')
        
        # Validation
        username_valid, username_msg = validate_username(username)
        if not username_valid:
            flash(username_msg, 'danger')
            return render_template('signup.html')
        
        if not validate_email(email):
            flash('Invalid email format', 'danger')
            return render_template('signup.html')
        
        password_valid, password_msg = validate_password(password)
        if not password_valid:
            flash(password_msg, 'danger')
            return render_template('signup.html')
        
        try:
            user_id = db.create_user(username, email, password, full_name, role)
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error creating account: {str(e)}', 'danger')
    
    return render_template('signup.html')

@app.route('/logout', methods=['POST'])
def logout():
    """Logout"""
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard page"""
    try:
        stats = db.get_dashboard_stats(session['user_id'])
        current_time = datetime.now().strftime('%A, %B %d, %Y')
        
        return render_template('dashboard.html', stats=stats, current_time=current_time, session=session)
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading dashboard', 'danger')
        return redirect(url_for('login'))

@app.route('/patients')
@login_required
def patients():
    """Patients list page"""
    try:
        patients = db.get_patients_by_user(session['user_id'])
        return render_template('patients.html', patients=patients)
    except Exception as e:
        print(f"Patients list error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading patients', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/health-history')
@login_required
def health_history():
    """Health check history page"""
    try:
        # Get all predictions for this user's patients
        predictions = db.get_recent_predictions(session['user_id'], limit=100)
        patients = db.get_patients_by_user(session['user_id'])
        
        # Create a patient lookup dict using 'id' field
        patient_dict = {str(p.get('id')): p for p in patients}
        
        # Add patient info and calculate confidence percentage
        for pred in predictions:
            patient_id = str(pred.get('patient_id'))
            patient = patient_dict.get(patient_id, {})
            pred['patient'] = {
                'name': patient.get('name', patient.get('full_name', 'Unknown')),
                'patient_id': patient.get('patient_id', 'N/A')
            }
            # Calculate confidence percentage for template
            if pred.get('confidence'):
                pred['confidence_pct'] = pred['confidence'] * 100
            else:
                pred['confidence_pct'] = 0
        
        return render_template('health_history.html', predictions=predictions)
    except Exception as e:
        print(f"Health history error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading health history', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/analytics')
@login_required
def analytics():
    """Analytics and trends page"""
    try:
        analytics_data = db.get_analytics_data(session['user_id'])
        return render_template('analytics.html', analytics_data=analytics_data)
    except Exception as e:
        print(f"Analytics error: {str(e)}")
        import traceback
        traceback.print_exc()
        flash('Error loading analytics', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/patient/<patient_id>')
@login_required
def patient_detail(patient_id):
    """Patient detail page"""
    try:
        patient = db.get_patient_by_id(patient_id)
        if not patient:
            flash('Patient not found', 'danger')
            return redirect(url_for('patients'))
        
        predictions = db.get_patient_predictions(patient_id)
        return render_template('patient_detail.html', patient=patient, predictions=predictions)
    except Exception as e:
        print(f"Error in patient_detail: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error loading patient details: {str(e)}', 'danger')
        return redirect(url_for('patients'))

@app.route('/export-report/<patient_id>')
@login_required
def export_report(patient_id):
    """Export patient health report as PDF"""
    try:
        # Get patient data
        patient = db.get_patient_by_id(patient_id)
        if not patient:
            flash('Patient not found', 'danger')
            return redirect(url_for('patients'))
        
        # Get all predictions for this patient
        predictions = db.get_patient_predictions(patient_id)
        
        # Generate PDF filename
        patient_name = patient.get('name', patient.get('full_name', 'patient')).replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"health_report_{patient_name}_{timestamp}.pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Generate PDF
        generate_patient_report(patient, predictions, pdf_path)
        
        # Send file for download
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        import traceback
        traceback.print_exc()
        flash('Error generating report. Please try again.', 'danger')
        return redirect(url_for('patient_detail', patient_id=patient_id))

@app.route('/add-patient', methods=['GET', 'POST'])
@login_required
def add_patient():
    """Add new patient"""
    if request.method == 'POST':
        try:
            patient_data = {
                'patient_id': request.form.get('patient_id'),
                'full_name': request.form.get('full_name'),
                'age': request.form.get('age'),
                'gender': request.form.get('gender'),
                'phone': request.form.get('phone'),
                'email': request.form.get('email'),
                'address': request.form.get('address'),
                'medical_history': request.form.get('medical_history'),
                'diabetes_type': request.form.get('diabetes_type'),
                'diabetes_duration': request.form.get('diabetes_duration'),
                'blood_sugar_level': request.form.get('blood_sugar_level'),
                'hba1c_level': request.form.get('hba1c_level'),
                'has_diabetes': request.form.get('has_diabetes', 'No')
            }
            
            print(f"Creating patient with data: {patient_data}")
            print(f"User ID: {session['user_id']}")
            
            patient_id = db.create_patient(patient_data, session['user_id'])
            print(f"Patient created with ID: {patient_id}")
            
            flash('Family member added successfully!', 'success')
            return redirect(url_for('patient_detail', patient_id=patient_id))
        except Exception as e:
            print(f"Add patient error: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'Error adding family member: {str(e)}', 'danger')
            return redirect(url_for('add_patient'))
    
    generated_patient_id = generate_patient_id()
    return render_template('add_patient.html', generated_patient_id=generated_patient_id)

@app.route('/dfu-detection')
@login_required
def dfu_detection():
    """DFU Detection page"""
    # Get all patients for the current user
    patients = db.get_patients_by_user(session['user_id'])
    patient_id = request.args.get('patient_id')
    patient = None
    if patient_id:
        patient = db.get_patient_by_id(patient_id)
    return render_template('dfu_detection.html', patient=patient, patients=patients)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Prediction endpoint - now with patient tracking"""
    print("\n=== PREDICT ROUTE CALLED ===")
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file:
        result = predict_image(file)
        print(f"Prediction result: {result}")
        
        # Save prediction to database
        patient_id = request.form.get('patient_id')
        print(f"Patient ID from form: {patient_id}")
        
        if patient_id and result.get('success'):
            try:
                print("Attempting to save prediction...")
                # Save image
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                image_filename = f"{patient_id}_{timestamp}_{filename}"
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                file.seek(0)  # Reset file pointer
                file.save(image_path)
                
                # Save prediction
                prediction_data = result.copy()
                prediction_data['image_path'] = image_path
                prediction_data['notes'] = request.form.get('notes', '')
                
                prediction_id = db.save_prediction(patient_id, prediction_data)
                print(f"Prediction saved successfully! ID: {prediction_id}")
            except Exception as e:
                print(f"Error saving prediction: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Prediction NOT saved. patient_id: {patient_id}, success: {result.get('success')}")
        
        return jsonify(result)
    
    return jsonify({'success': False, 'error': 'Invalid file'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'not set'
    })

if __name__ == '__main__':
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("\n" + "="*80)
    print("DFU Detection System - Complete Medical Application")
    print("="*80)
    
    # Initialize database
    print("\nInitializing database...")
    try:
        db.init_database()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Database error: {e}")
    
    # Don't load model at startup - will load on first prediction
    print("\nModel will be loaded on first prediction request...")
    
    print("\n" + "="*80)
    print("Starting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Default login: username=admin, password=admin123")
    print("="*80 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"\nServer error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
