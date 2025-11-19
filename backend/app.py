
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import secrets
import string
import re
import datetime
from datetime import timezone
from functools import wraps

from dotenv import load_dotenv
from flask import (
    Flask, request, jsonify, send_from_directory, make_response
)
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import bcrypt
import jwt
import joblib
import numpy as np
import pandas as pd

# -------------------------
# Load environment
# -------------------------
load_dotenv()

# -------------------------
# App & logging setup
# -------------------------
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend/templates')

# -------------------------
# Load ML Models
# -------------------------
ML_MODEL = None
SCALER = None
LABEL_ENCODERS = None

def load_ml_models():
    """Load the trained ML model and preprocessing objects"""
    global ML_MODEL, SCALER, LABEL_ENCODERS
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model.pkl')
        scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scaler.pkl')
        encoders_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'label_encoders.pkl')
        
        ML_MODEL = joblib.load(model_path)
        SCALER = joblib.load(scaler_path)
        LABEL_ENCODERS = joblib.load(encoders_path)
        
        print("✅ ML models loaded successfully!")
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not load ML models: {e}")
        print("   Predictions will use fallback logic.")
        return False

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.getenv('LOG_FILE', os.path.join(LOG_DIR, 'edu2job.log'))
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
file_handler.setLevel(getattr(logging, log_level, logging.INFO))

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
console_handler.setLevel(getattr(logging, log_level, logging.INFO))

app.logger.addHandler(file_handler)
app.logger.addHandler(console_handler)
app.logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.basicConfig(level=getattr(logging, log_level, logging.INFO), handlers=[file_handler, console_handler])

logger = app.logger

# -------------------------
# Configuration
# -------------------------
SECRET_KEY = os.getenv('SECRET_KEY')
DATABASE_URL = os.getenv('DATABASE_URL')

if not SECRET_KEY:
    logger.warning("SECRET_KEY not set — generating a temporary key for this run.")
    SECRET_KEY = secrets.token_hex(32)
    logger.warning("Set SECRET_KEY in .env for persistence.")

if not DATABASE_URL:
    logger.critical("DATABASE_URL is not set. Please set DATABASE_URL in .env (e.g. sqlite:///database.db or mysql+pymysql://user:pass@host/db)")
    sys.exit(1)

app.config['SECRET_KEY'] = SECRET_KEY
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# JWT expiry settings (seconds)
ACCESS_TOKEN_EXPIRES = int(os.getenv('ACCESS_TOKEN_EXPIRES_SECONDS', 15 * 60))  # 15 minutes default
REFRESH_TOKEN_EXPIRES = int(os.getenv('REFRESH_TOKEN_EXPIRES_SECONDS', 7 * 24 * 3600))  # 7 days

# Session / cookie security flags (for production)
app.config['SESSION_COOKIE_HTTPONLY'] = True

# -------------------------
# Extensions
# -------------------------
db = SQLAlchemy(app)
CORS(app, resources={r"/*": {"origins": ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:*", "http://127.0.0.1:*"]}}, supports_credentials=True)
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

# -------------------------
# Helpers
# -------------------------
def utcnow():
    return datetime.datetime.now(timezone.utc)

def sanitize_input(text, max_length=1000):
    if text is None:
        return None
    text = re.sub(r'<[^>]*>', '', str(text))
    return text.strip()[:max_length]

def validate_email(email):
    if not email:
        return False
    email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(email_regex, email) is not None

def validate_password_strength(password):
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain a letter"
    if not re.search(r'\d', password):
        return False, "Password must contain a number"
    return True, "OK"

def generate_token(payload, expire_seconds):
    now = int(time.time())
    payload_copy = payload.copy()
    payload_copy['exp'] = now + int(expire_seconds)
    return jwt.encode(payload_copy, app.config['SECRET_KEY'], algorithm='HS256')

def decode_token(token):
    return jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])

def generate_reset_token(length=64):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# -------------------------
# Database connection test
# -------------------------
def test_database_connection(max_retries=5, delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            with app.app_context():
                db.engine.connect()
            logger.info("Database connection OK")
            return True
        except Exception as e:
            logger.warning(f"DB connect attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
    return False

if not test_database_connection():
    logger.critical("Cannot connect to database. Exiting.")
    sys.exit(1)

# Load ML models
load_ml_models()

# -------------------------
# Models
# -------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)

    # profile fields (optional)
    phone = db.Column(db.String(20))
    location = db.Column(db.String(100))
    headline = db.Column(db.String(200))
    summary = db.Column(db.Text)

    # relationships
    # Simplified for single-file version
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
        except Exception:
            return False

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    def __init__(self, username, password):
        self.username = username
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    def check_password(self, password):
        try:
            return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))
        except Exception:
            return False

class PasswordResetToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    token = db.Column(db.String(128), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    used = db.Column(db.Boolean, default=False)

    def is_valid(self):
        return (not self.used) and (utcnow() < self.expires_at)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    predicted_role = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    salary_range = db.Column(db.String(50))
    degree = db.Column(db.String(100))
    major = db.Column(db.String(100))
    specialization = db.Column(db.String(100))
    cgpa = db.Column(db.Float)
    years_of_experience = db.Column(db.Integer)
    skills = db.Column(db.Text)
    certifications = db.Column(db.Text)
    preferred_industry = db.Column(db.String(100))
    created_at = db.Column(db.DateTime, default=utcnow)
    
    def to_dict(self):
        """Convert to dictionary for JSON response"""
        # Calculate time ago
        now = utcnow()
        delta = now - self.created_at
        
        if delta.days > 365:
            years = delta.days // 365
            time_ago = f"{years} year{'s' if years > 1 else ''} ago"
        elif delta.days > 30:
            months = delta.days // 30
            time_ago = f"{months} month{'s' if months > 1 else ''} ago"
        elif delta.days > 0:
            time_ago = f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            time_ago = f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            time_ago = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            time_ago = "Just now"
        
        # Create search title from input data
        search_title = f"{self.degree} in {self.major}"
        if self.specialization:
            search_title += f" - {self.specialization}"
        
        return {
            'id': self.id,
            'predicted_role': self.predicted_role,
            'confidence': round(self.confidence, 1),
            'salary_range': self.salary_range,
            'search_title': search_title,
            'time_ago': time_ago,
            'created_at': self.created_at.isoformat(),
            'input_data': {
                'degree': self.degree,
                'major': self.major,
                'specialization': self.specialization,
                'cgpa': self.cgpa,
                'years_of_experience': self.years_of_experience,
                'skills': self.skills,
                'certifications': self.certifications,
                'preferred_industry': self.preferred_industry
            }
        }

# create tables and default admin
with app.app_context():
    db.create_all()
    if not Admin.query.filter_by(username='admin').first():
        admin = Admin('admin', 'admin123')
        db.session.add(admin)
        db.session.commit()
        logger.info("Default admin created (admin/admin123)")

# -------------------------
# Auth decorators
# -------------------------
def token_from_request():
    token = request.cookies.get('access_token')
    if not token and 'Authorization' in request.headers:
        parts = request.headers.get('Authorization').split()
        if len(parts) == 2 and parts[0] == 'Bearer':
            token = parts[1]
    return token

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = token_from_request()
        if not token:
            return jsonify({"message": "Token missing"}), 401
        try:
            decoded = decode_token(token)
            user = User.query.get(decoded.get('user_id'))
            if not user:
                return jsonify({"message": "User not found"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            logger.error(f"Token decode error: {e}")
            return jsonify({"message": "Token error"}), 401
        return f(user, *args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = token_from_request()
        if not token:
            return jsonify({"message": "Admin token missing"}), 401
        try:
            decoded = decode_token(token)
            if decoded.get('role') != 'admin':
                return jsonify({"message": "Admin role required"}), 403
            admin = Admin.query.get(decoded.get('admin_id'))
            if not admin:
                return jsonify({"message": "Admin not found"}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Admin token expired"}), 401
        except Exception as e:
            logger.error(f"Admin token error: {e}")
            return jsonify({"message": "Admin token invalid"}), 401
        return f(admin, *args, **kwargs)
    return wrapper

# -------------------------
# Routes: Auth
# -------------------------
@app.route('/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    data = request.get_json() or {}
    username = sanitize_input(data.get('username', ''))
    email = (data.get('email') or '').strip().lower()
    password = data.get('password', '')

    if not username or not email or not password:
        return jsonify({"message": "All fields required"}), 400
    if not validate_email(email):
        return jsonify({"message": "Invalid email"}), 400
    ok, msg = validate_password_strength(password)
    if not ok:
        return jsonify({"message": msg}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"message": "Email already registered"}), 409
    user = User(username, email, password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered", "username": user.username, "email": user.email}), 201

@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({"message": "Email and password required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"message": "Invalid credentials"}), 401

    access_payload = {"user_id": user.id, "type": "access"}
    refresh_payload = {"user_id": user.id, "type": "refresh"}

    access_token = generate_token(access_payload, ACCESS_TOKEN_EXPIRES)
    refresh_token = generate_token(refresh_payload, REFRESH_TOKEN_EXPIRES)

    resp = make_response(jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": {"id": user.id, "username": user.username, "email": user.email}
    }))
    # Set HttpOnly cookies
    resp.set_cookie('access_token', access_token, httponly=True, samesite='Lax', max_age=ACCESS_TOKEN_EXPIRES)
    resp.set_cookie('refresh_token', refresh_token, httponly=True, samesite='Lax', max_age=REFRESH_TOKEN_EXPIRES)
    return resp, 200

@app.route('/logout', methods=['POST'])
def logout():
    resp = make_response(jsonify({"message": "Logged out"}))
    resp.set_cookie('access_token', '', expires=0)
    resp.set_cookie('refresh_token', '', expires=0)
    return resp, 200

@app.route('/verify-token', methods=['GET'])
def verify_token_route():
    token = token_from_request()
    if not token:
        return jsonify({"valid": False, "message": "No token provided"}), 401
    try:
        decoded = decode_token(token)
        user = User.query.get(decoded.get('user_id'))
        if not user:
            return jsonify({"valid": False, "message": "User not found"}), 401
        return jsonify({"valid": True, "user": {"id": user.id, "email": user.email}}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({"valid": False, "message": "Token expired"}), 401
    except Exception as e:
        logger.warning(f"Token verify error: {e}")
        return jsonify({"valid": False, "message": "Invalid token"}), 401

# -------------------------
# Profile & Admin routes
# -------------------------
@app.route('/api/profile', methods=['GET'])
@login_required
def get_profile(user):
    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "phone": user.phone,
        "location": user.location,
        "headline": user.headline,
        "summary": user.summary
    }), 200

@app.route('/api/profile', methods=['PUT'])
@login_required
def update_profile(user):
    data = request.get_json() or {}
    user.phone = sanitize_input(data.get('phone', user.phone))
    user.location = sanitize_input(data.get('location', user.location))
    user.headline = sanitize_input(data.get('headline', user.headline), max_length=200)
    user.summary = sanitize_input(data.get('summary', user.summary), max_length=2000)
    db.session.commit()
    return jsonify({"message": "Profile updated"}), 200

@app.route('/api/users', methods=['GET'])
@admin_required
def admin_get_users(admin):
    users = User.query.all()
    return jsonify([{"id": u.id, "username": u.username, "email": u.email} for u in users]), 200

@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json() or {}
    username = data.get('username', '')
    password = data.get('password', '')
    admin = Admin.query.filter_by(username=username).first()
    if not admin or not admin.check_password(password):
        return jsonify({"message": "Invalid admin credentials"}), 401
    payload = {"admin_id": admin.id, "role": "admin"}
    token = generate_token(payload, ACCESS_TOKEN_EXPIRES * 4)  # longer admin token
    return jsonify({"token": token, "message": "Admin login successful"}), 200

@app.route('/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def admin_delete_user(admin, user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({"message": "User not found"}), 404
    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted"}), 200

# -------------------------
# Password reset flow
# -------------------------
@app.route('/forgot-password', methods=['POST'])
@limiter.limit("3 per hour")
def forgot_password():
    data = request.get_json() or {}
    email = (data.get('email') or '').strip().lower()
    if not email or not validate_email(email):
        return jsonify({"message": "Invalid email"}), 400
    user = User.query.filter_by(email=email).first()
    # Always return same response to avoid enumeration
    if not user:
        return jsonify({"message": "If account exists, reset instructions sent"}), 200
    token = generate_reset_token()
    expires_at = utcnow() + datetime.timedelta(hours=1)
    # invalidate previous tokens
    PasswordResetToken.query.filter_by(user_id=user.id, used=False).update({'used': True})
    reset = PasswordResetToken(user_id=user.id, token=token, expires_at=expires_at)
    db.session.add(reset)
    db.session.commit()
    logger.info(f"Password reset token for {email} (dev only): {token}")
    return jsonify({"message": "If account exists, reset instructions sent", "token": token}), 200

@app.route('/reset-password', methods=['POST'])
@limiter.limit("5 per hour")
def reset_password():
    data = request.get_json() or {}
    token = data.get('token', '').strip()
    new_password = data.get('new_password', '')
    if not token or not new_password:
        return jsonify({"message": "Token and new password required"}), 400
    ok, msg = validate_password_strength(new_password)
    if not ok:
        return jsonify({"message": msg}), 400
    reset = PasswordResetToken.query.filter_by(token=token).first()
    if not reset or not reset.is_valid():
        return jsonify({"message": "Invalid or expired token"}), 400
    user = User.query.get(reset.user_id)
    if not user:
        return jsonify({"message": "User not found"}), 404
    user.password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    reset.used = True
    db.session.commit()
    return jsonify({"message": "Password reset successful"}), 200

# -------------------------
# Prediction with ML Model
# -------------------------

def normalize_cgpa_scale(cgpa):
    """
    Normalize CGPA to 10-point scale.
    Detects if GPA is on 4-point scale and converts it.
    """
    cgpa_float = float(cgpa)
    
    # If CGPA is between 0 and 4.5, assume it's US GPA (4-point scale)
    if 0 < cgpa_float <= 4.5:
        # Convert 4-point scale to 10-point scale
        # Simple linear conversion: (GPA / 4) * 10
        normalized = (cgpa_float / 4.0) * 10.0
        logger.info(f"Converted US GPA {cgpa_float} to 10-point scale: {normalized:.2f}")
        return normalized
    
    # If between 4.5 and 10, assume it's already on 10-point scale
    elif 4.5 < cgpa_float <= 10.0:
        return cgpa_float
    
    # If greater than 10, might be percentage - convert
    elif cgpa_float > 10:
        # Assume it's percentage, convert to 10-point scale
        normalized = cgpa_float / 10.0
        logger.info(f"Converted percentage {cgpa_float} to 10-point scale: {normalized:.2f}")
        return min(normalized, 10.0)
    
    else:
        return cgpa_float

def calculate_string_similarity(str1, str2):
    """
    Calculate similarity between two strings using character-level matching.
    Returns a score between 0 and 1.
    """
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    # Exact match
    if str1_lower == str2_lower:
        return 1.0
    
    # Check if one contains the other
    if str1_lower in str2_lower or str2_lower in str1_lower:
        return 0.8
    
    # Calculate Jaccard similarity on character bigrams
    def get_bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    
    bigrams1 = get_bigrams(str1_lower)
    bigrams2 = get_bigrams(str2_lower)
    
    if not bigrams1 or not bigrams2:
        return 0.0
    
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    return intersection / union if union > 0 else 0.0

def find_closest_match(value, encoder, col_name):
    """
    Find the closest match for a value in the encoder's classes.
    Uses improved fuzzy matching with better string similarity.
    """
    classes = encoder.classes_
    
    # Handle empty/None values
    if not value or value.strip() == '':
        return None
    
    value = value.strip()
    
    # Exact match (case-sensitive)
    if value in classes:
        return value
    
    # Case-insensitive exact match
    for cls in classes:
        if cls.lower() == value.lower():
            return cls
    
    # Fuzzy matching for all fields
    best_match = None
    best_score = 0.0
    
    # For Skills and Certification, use keyword-based matching
    if col_name in ['Skills', 'Certification']:
        # Normalize input
        value_lower = value.lower()
        value_keywords = set(re.sub(r'[^a-z0-9\s]', ' ', value_lower).split())
        value_keywords = {k for k in value_keywords if len(k) > 1}  # Remove single chars
        
        for cls in classes:
            cls_lower = cls.lower()
            cls_keywords = set(re.sub(r'[^a-z0-9\s]', ' ', cls_lower).split())
            cls_keywords = {k for k in cls_keywords if len(k) > 1}
            
            if not value_keywords or not cls_keywords:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(value_keywords & cls_keywords)
            union = len(value_keywords | cls_keywords)
            score = intersection / union if union > 0 else 0.0
            
            # Also consider string similarity
            str_similarity = calculate_string_similarity(value, cls)
            
            # Combined score (weighted average)
            combined_score = (score * 0.7) + (str_similarity * 0.3)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match = cls
    else:
        # For other fields, use string similarity
        for cls in classes:
            score = calculate_string_similarity(value, cls)
            
            if score > best_score:
                best_score = score
                best_match = cls
    
    # Return best match only if score is good enough
    # Increased threshold from 0.3 to 0.5 for better accuracy
    if best_match and best_score >= 0.5:
        if best_match != value:
            logger.info(f"Fuzzy match for {col_name}: '{value}' -> '{best_match}' (score: {best_score:.2f})")
        return best_match
    
    # No good match found
    logger.warning(f"No good match found for {col_name}: '{value}' (best score: {best_score:.2f})")
    return None

def preprocess_input_for_prediction(degree, major, specialization, cgpa, years_of_experience, skills, certifications, preferred_industry):
    """
    Preprocess user input to match the format expected by the ML model.
    CRITICAL: Must match the exact preprocessing pipeline used in generate_model.py
    Returns a pandas DataFrame ready for prediction.
    """
    try:
        # Normalize CGPA scale (convert US GPA to 10-point scale if needed)
        normalized_cgpa = normalize_cgpa_scale(cgpa)
        
        # Create a dictionary with the input data (exactly as in training)
        # IMPORTANT: Keep original numerical values for scaling
        input_data = {
            'Degree': degree.strip(),
            'Major': major.strip(),
            'Specialization': specialization.strip() if specialization else '',
            'CGPA': normalized_cgpa,  # Use normalized value
            'Skills': skills.strip(),
            'Certification': certifications.strip() if certifications else 'None',
            'Years of Experience': float(years_of_experience),
            'Preferred Industry': preferred_industry.strip()
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # CRITICAL FIX: Scale numerical features BEFORE encoding
        # This matches the training pipeline in generate_model.py (lines 57-59)
        # The scaler was fitted on RAW CGPA and Experience values, not encoded ones
        numerical_cols = ['CGPA', 'Years of Experience']
        df[numerical_cols] = SCALER.transform(df[numerical_cols])
        
        logger.info(f"Scaled values - CGPA: {df['CGPA'].iloc[0]:.4f}, Experience: {df['Years of Experience'].iloc[0]:.4f}")
        
        # Store scaled values
        scaled_cgpa = df['CGPA'].iloc[0]
        scaled_exp = df['Years of Experience'].iloc[0]
        
        # Now encode categorical variables using the trained label encoders
        categorical_cols = ['Degree', 'Major', 'Specialization', 'Skills', 'Certification', 'Preferred Industry']
        
        match_warnings = []
        
        for col in categorical_cols:
            if col in LABEL_ENCODERS:
                le = LABEL_ENCODERS[col]
                value = input_data[col]  # Use original string value
                
                # Try to find exact or close match
                matched_value = find_closest_match(value, le, col)
                
                if matched_value:
                    df[col] = le.transform([matched_value])[0]
                    if matched_value != value:
                        match_warnings.append(f"{col}: '{value}' matched to '{matched_value}'")
                else:
                    # Use the most common class (middle of sorted classes) instead of 0
                    # This is more statistically sound than always using the first class
                    classes = le.classes_
                    default_idx = len(classes) // 2  # Use middle class as default
                    default_class = classes[default_idx]
                    df[col] = default_idx
                    logger.warning(f"No match found for {col}: '{value}'. Using default: '{default_class}' (index {default_idx})")
                    match_warnings.append(f"{col}: '{value}' had no match, using default '{default_class}'")
        
        # Log all match warnings at once
        if match_warnings:
            logger.info("Input matching results:\n  " + "\n  ".join(match_warnings))
        
        # Put scaled values back (they were overwritten during categorical processing)
        df['CGPA'] = scaled_cgpa
        df['Years of Experience'] = scaled_exp
        
        # Ensure correct column order to match training
        # This should match the order in generate_model.py after encoding and scaling
        column_order = ['Degree', 'Major', 'Specialization', 'CGPA', 'Skills', 'Certification', 'Years of Experience', 'Preferred Industry']
        df = df[column_order]
        
        logger.info(f"Final feature vector shape: {df.shape}")
        logger.debug(f"Final feature values: {df.values[0]}")
        
        return df
    except Exception as e:
        logger.error(f"Error preprocessing input: {e}", exc_info=True)
        raise

def generate_job_predictions_ml(degree, major, specialization, cgpa, years_of_experience, skills, certifications, preferred_industry):
    """
    Generate job predictions using the trained ML model.
    Falls back to rule-based predictions if ML model is not available.
    """
    if ML_MODEL is None or SCALER is None or LABEL_ENCODERS is None:
        logger.warning("ML model not loaded, using fallback predictions")
        return generate_job_predictions_fallback(degree, major, specialization, cgpa, years_of_experience, skills, certifications, preferred_industry)
    
    try:
        # Preprocess input
        input_df = preprocess_input_for_prediction(
            degree, major, specialization, cgpa, years_of_experience, 
            skills, certifications, preferred_industry
        )
        
        logger.info(f"Preprocessed input shape: {input_df.shape}, columns: {input_df.columns.tolist()}")
        
        # Get prediction probabilities
        probabilities = ML_MODEL.predict_proba(input_df)[0]
        
        # Get top 5 predictions with highest probability
        top_indices = np.argsort(probabilities)[-5:][::-1]
        
        # Decode job role labels
        job_role_encoder = LABEL_ENCODERS['Job Role']
        
        predictions = []
        for idx in top_indices:
            confidence = probabilities[idx] * 100
            # Show predictions with reasonable confidence
            if confidence > 1:  # Lower threshold to show more options
                job_role = job_role_encoder.inverse_transform([idx])[0]
                predictions.append({
                    'job_role': job_role,
                    'confidence': round(confidence, 1),
                    'salary': estimate_salary(job_role, years_of_experience, cgpa)
                })
        
        # If we don't have enough predictions, ensure we have at least top prediction
        if len(predictions) == 0:
            top_idx = np.argmax(probabilities)
            job_role = job_role_encoder.inverse_transform([top_idx])[0]
            predictions.append({
                'job_role': job_role,
                'confidence': round(probabilities[top_idx] * 100, 1),
                'salary': estimate_salary(job_role, years_of_experience, cgpa),
                'details': {
                    'salary_range': estimate_salary(job_role, years_of_experience, cgpa)
                }
            })
        else:
            # Add detailed info to all predictions
            for pred in predictions:
                if 'details' not in pred:
                    pred['details'] = {
                        'salary_range': pred.get('salary', estimate_salary(pred['job_role'], years_of_experience, cgpa))
                    }
        
        logger.info(f"Generated {len(predictions)} predictions. Top: {predictions[0]['job_role']} ({predictions[0]['confidence']}%)")
        return predictions
    except Exception as e:
        logger.error(f"Error in ML prediction: {e}", exc_info=True)
        # Fall back to rule-based predictions
        return generate_job_predictions_fallback(degree, major, specialization, cgpa, years_of_experience, skills, certifications, preferred_industry)

def estimate_salary(job_role, years_of_experience, cgpa):
    """
    Estimate salary range based on job role, experience, and academic performance.
    Returns salary in Indian Rupees (LPA - Lakhs Per Annum).
    """
    # Base salary ranges (in LPA - Lakhs Per Annum) for entry-level to mid-level
    salary_ranges = {
        'Software Engineer': (6, 18),
        'Data Scientist': (8, 25),
        'Full Stack Developer': (6, 20),
        'Machine Learning Engineer': (10, 30),
        'DevOps Engineer': (7, 22),
        'Product Manager': (12, 35),
        'Business Analyst': (5, 15),
        'Data Analyst': (5, 15),
        'Web Developer': (4, 15),
        'Mobile App Developer': (6, 18),
        'Cloud Architect': (15, 40),
        'Cybersecurity Analyst': (8, 25),
        'AI Engineer': (12, 35),
        'Backend Developer': (6, 20),
        'Frontend Developer': (5, 18),
        'UI/UX Designer': (5, 16),
        'QA Engineer': (4, 14),
        'System Administrator': (5, 15),
        'Database Administrator': (7, 20),
        'Network Engineer': (6, 18),
    }
    
    # Get base salary or use default
    base_min, base_max = salary_ranges.get(job_role, (5, 15))
    
    # Adjust based on experience (non-linear growth)
    # Early years contribute more to salary growth
    if years_of_experience <= 2:
        exp_multiplier = 1 + (years_of_experience * 0.15)  # 15% per year
    elif years_of_experience <= 5:
        exp_multiplier = 1.3 + ((years_of_experience - 2) * 0.12)  # 12% per year
    else:
        exp_multiplier = 1.66 + ((years_of_experience - 5) * 0.08)  # 8% per year (capped growth)
    
    # Adjust based on CGPA (on 10-point scale)
    # Higher CGPA indicates better potential, especially for freshers
    if cgpa >= 8.5:
        cgpa_multiplier = 1.15  # 15% boost for excellent grades
    elif cgpa >= 7.5:
        cgpa_multiplier = 1.08  # 8% boost for good grades
    elif cgpa >= 6.5:
        cgpa_multiplier = 1.00  # No adjustment
    else:
        cgpa_multiplier = 0.95  # 5% reduction for lower grades
    
    # Calculate adjusted salary
    adjusted_min = int(base_min * exp_multiplier * cgpa_multiplier)
    adjusted_max = int(base_max * exp_multiplier * cgpa_multiplier)
    
    # Ensure minimum salary
    adjusted_min = max(adjusted_min, 3)
    adjusted_max = max(adjusted_max, adjusted_min + 3)
    
    return f"₹{adjusted_min}-{adjusted_max} LPA"

def generate_job_predictions_fallback(degree, major, specialization, cgpa, years_of_experience, skills, certifications, preferred_industry):
    """Fallback rule-based prediction when ML model is not available"""
    predictions = []
    job_roles = {
        "Software Engineer": {"majors":["computer science","software engineering","information technology"], "skills":["python","java","javascript","c++","git","sql"], "min_cgpa":6.5, "salary":"₹6-18 LPA"},
        "Data Scientist": {"majors":["data science","computer science","statistics"], "skills":["python","machine learning","sql","r","statistics"], "min_cgpa":7.0, "salary":"₹8-25 LPA"},
        "Full Stack Developer": {"majors":["computer science","software engineering"], "skills":["javascript","react","node","html","css"], "min_cgpa":6.0, "salary":"₹6-20 LPA"},
    }
    skills_lower = [s.lower() for s in skills.split(',')]
    major_l = major.lower()
    for role, req in job_roles.items():
        score = 0
        if any(m in major_l for m in req["majors"]):
            score += 30
        matched = sum(1 for s in req["skills"] if any(s in sk for sk in skills_lower))
        score += min(30, matched * 10)
        if cgpa >= req["min_cgpa"]:
            score += 20
        if years_of_experience >= 2:
            score += 10
        confidence = min(100, score)
        if confidence > 30:
            predictions.append({"job_role": role, "confidence": confidence, "salary": req["salary"]})
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions[:5]

@app.route('/api/get-options', methods=['GET'])
def get_options():
    """
    Get available options for all categorical fields from the trained model.
    This helps frontend provide accurate dropdowns to users.
    Returns EXACT values from training data to ensure perfect matches.
    """
    if LABEL_ENCODERS is None:
        return jsonify({"message": "Model not loaded"}), 503
    
    try:
        options = {}
        
        # Return ALL actual values from the label encoders
        # These are the ONLY values that will match during prediction
        for col in ['Degree', 'Major', 'Specialization', 'Preferred Industry']:
            if col in LABEL_ENCODERS:
                # Return sorted list of ALL possible values
                options[col] = sorted(LABEL_ENCODERS[col].classes_.tolist())
        
        # For Skills and Certification, return ALL options
        # Users need to see what exact skill combinations exist in training data
        if 'Skills' in LABEL_ENCODERS:
            all_skills = LABEL_ENCODERS['Skills'].classes_.tolist()
            options['Skills'] = sorted(all_skills)
            
            # Also provide top 30 most common skills for quick reference
            # Filter for common tech skills
            common_keywords = ['Python', 'Java', 'HTML', 'CSS', 'JavaScript', 'SQL', 'React', 
                             'Node', 'MongoDB', 'PHP', 'MySQL', 'Bootstrap', 'Vue', 'Express',
                             'Angular', 'Django', 'Flask', 'AWS', 'Docker', 'Git']
            options['Skills_popular'] = [s for s in all_skills if any(kw in s for kw in common_keywords)][:30]
        
        if 'Certification' in LABEL_ENCODERS:
            all_certs = LABEL_ENCODERS['Certification'].classes_.tolist()
            options['Certification'] = sorted(all_certs)
            
            # Provide popular certifications
            popular_cert_keywords = ['AWS', 'Google', 'Microsoft', 'Coursera', 'NPTEL', 
                                    'HackerRank', 'Udemy', 'Infosys', 'None']
            options['Certification_popular'] = [c for c in all_certs if any(kw in c for kw in popular_cert_keywords)][:30]
        
        # Add statistics to help users understand the data
        options['stats'] = {
            'total_degrees': len(LABEL_ENCODERS['Degree'].classes_) if 'Degree' in LABEL_ENCODERS else 0,
            'total_majors': len(LABEL_ENCODERS['Major'].classes_) if 'Major' in LABEL_ENCODERS else 0,
            'total_skills': len(LABEL_ENCODERS['Skills'].classes_) if 'Skills' in LABEL_ENCODERS else 0,
            'total_certifications': len(LABEL_ENCODERS['Certification'].classes_) if 'Certification' in LABEL_ENCODERS else 0,
            'total_industries': len(LABEL_ENCODERS['Preferred Industry'].classes_) if 'Preferred Industry' in LABEL_ENCODERS else 0,
            'total_job_roles': len(LABEL_ENCODERS['Job Role'].classes_) if 'Job Role' in LABEL_ENCODERS else 0
        }
        
        logger.info(f"Returning options with {options['stats']} items")
        
        return jsonify(options), 200
    except Exception as e:
        logger.error(f"Error getting options: {e}", exc_info=True)
        return jsonify({"message": "Error loading options"}), 500

@app.route('/api/prediction-history', methods=['GET'])
@login_required
def get_prediction_history(user):
    """Get user's prediction history, ordered by most recent first"""
    try:
        # Get all predictions for this user, ordered by newest first
        predictions = PredictionHistory.query.filter_by(user_id=user.id).order_by(PredictionHistory.created_at.desc()).limit(50).all()
        
        # Convert to dictionary format
        history = [pred.to_dict() for pred in predictions]
        
        logger.info(f"Retrieved {len(history)} history items for user {user.id}")
        
        return jsonify({"history": history}), 200
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {e}", exc_info=True)
        return jsonify({"message": "Error loading prediction history", "history": []}), 500

@app.route('/api/predict-job', methods=['POST'])
@login_required
def predict_job(user):
    data = request.get_json() or {}
    degree = (data.get('degree') or '').strip()
    major = (data.get('major') or '').strip()
    specialization = (data.get('specialization') or '').strip()
    cgpa = data.get('cgpa')
    years_of_experience = data.get('years_of_experience')
    skills = (data.get('skills') or '').strip()  # Keep as comma-separated string
    certifications = (data.get('certifications') or '').strip()  # Keep as comma-separated string
    preferred_industry = (data.get('preferred_industry') or '').strip()

    # Basic validation
    if not degree or not major or not preferred_industry or not skills:
        return jsonify({"message": "Missing required fields"}), 400
    
    try:
        cgpa = float(cgpa)
        if cgpa < 0 or cgpa > 10:
            return jsonify({"message": "CGPA must be between 0 and 10"}), 400
    except Exception:
        return jsonify({"message": "Invalid CGPA"}), 400
    
    try:
        years_of_experience = int(years_of_experience)
        if years_of_experience < 0:
            return jsonify({"message": "Experience cannot be negative"}), 400
    except Exception:
        return jsonify({"message": "Invalid experience"}), 400

    # Get predictions using ML model
    preds = generate_job_predictions_ml(
        degree, major, specialization, cgpa, years_of_experience, 
        skills, certifications, preferred_industry
    )

    if not preds:
        return jsonify({"message": "Could not generate predictions. Please try different inputs."}), 200

    # Save top prediction to history
    try:
        top_prediction = preds[0]
        history_entry = PredictionHistory(
            user_id=user.id,
            predicted_role=top_prediction['job_role'],
            confidence=top_prediction['confidence'],
            salary_range=top_prediction.get('salary', ''),
            degree=degree,
            major=major,
            specialization=specialization,
            cgpa=cgpa,
            years_of_experience=years_of_experience,
            skills=skills,
            certifications=certifications,
            preferred_industry=preferred_industry
        )
        db.session.add(history_entry)
        db.session.commit()
        logger.info(f"Saved prediction to history for user {user.id}: {top_prediction['job_role']}")
    except Exception as e:
        logger.error(f"Error saving prediction to history: {e}")
        # Don't fail the request if history save fails
        db.session.rollback()

    return jsonify({"predictions": preds}), 200

# -------------------------
# Static routes for frontend
# -------------------------
@app.route('/')
def index_page():
    return send_from_directory('../frontend', 'login.html')

@app.route('/<path:filename>')
def serve_frontend(filename):
    return send_from_directory('../frontend', filename)

# -------------------------
# Run
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)), debug=(os.getenv('FLASK_ENV') != 'production'))
