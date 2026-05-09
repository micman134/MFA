"""
MFA System - AI Risk Assessment Service
AUTO-TRAINING VERSION - WITH CORRECT COLUMN NAMES
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import mysql.connector
from mysql.connector import Error
import joblib
import os
import hashlib
from datetime import datetime
import time
import threading
import json
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn loaded")
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ Scikit-learn not available - install with: pip install scikit-learn")

app = Flask(__name__)
CORS(app)

# Configuration
PORT = 5000
MODEL_PATH = 'models/risk_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ISOLATION_FOREST_PATH = 'models/isolation_forest.pkl'
TRAINING_LOG = 'models/training_log.json'

# Create models directory
os.makedirs('models', exist_ok=True)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'mfa_system',
    'user': 'root',
    'password': '',
    'port': 3306
}

# ============================================
# DATABASE CONNECTION
# ============================================
def get_db_connection():
    """Establish connection to MySQL database"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"❌ Database connection error: {e}")
        return None

# ============================================
# AUTO-TRAINING FUNCTION
# ============================================
def auto_train():
    """Automatically fetch data and train model on startup"""
    print("\n" + "="*60)
    print("🤖 AUTO-TRAINING MODE ACTIVE")
    print("="*60)
    
    # Check if scikit-learn is available
    if not SKLEARN_AVAILABLE:
        print("⚠️ Scikit-learn not installed. Cannot train model.")
        print("   Run: pip install scikit-learn")
        return
    
    # Check database connection
    print("\n🔌 Checking database connection...")
    conn = get_db_connection()
    if not conn:
        print("❌ Cannot connect to database. Auto-training skipped.")
        return
    
    print("✅ Database connected")
    conn.close()
    
    # Fetch training data
    print("\n📚 Fetching training data from database...")
    training_data = fetch_training_data()
    
    if not training_data or len(training_data) < 20:
        print(f"⚠️ Insufficient data: {len(training_data) if training_data else 0} records")
        print("   Need at least 20 records for training.")
        print("   Continuing with rule-based engine only.")
        return
    
    print(f"✅ Fetched {len(training_data)} records")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X, y = prepare_features(training_data)
    print(f"   Feature matrix: {X.shape}")
    print(f"   Risk score range: {y.min()*100:.1f} - {y.max()*100:.1f}")
    
    # Train ensemble models
    print("\n🧠 Training ensemble models...")
    models, scaler, metrics = train_ensemble_models(X, y)
    
    # Save models
    print("\n💾 Saving models...")
    joblib.dump(models['random_forest'], MODEL_PATH.replace('.pkl', '_rf.pkl'))
    joblib.dump(models['gradient_boosting'], MODEL_PATH.replace('.pkl', '_gb.pkl'))
    joblib.dump(models['isolation_forest'], ISOLATION_FOREST_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    # Save training info
    training_info = {
        'timestamp': datetime.now().isoformat(),
        'samples': len(X),
        'features': X.shape[1],
        'metrics': metrics,
        'model_types': ['RandomForest', 'GradientBoosting', 'IsolationForest']
    }
    
    with open(TRAINING_LOG, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print("\n✅ Auto-training completed successfully!")
    print(f"   Random Forest saved to: {MODEL_PATH.replace('.pkl', '_rf.pkl')}")
    print(f"   Gradient Boosting saved to: {MODEL_PATH.replace('.pkl', '_gb.pkl')}")
    print(f"   Isolation Forest saved to: {ISOLATION_FOREST_PATH}")
    print(f"   Training log: {TRAINING_LOG}")
    print("="*60 + "\n")
    
    # Reload model in risk engine
    global risk_engine
    risk_engine.load_model()

def fetch_training_data(limit=10000):
    """Fetch historical login data for training - Using your actual columns"""
    connection = get_db_connection()
    if not connection:
        return None
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Using your actual column names
        query = """
        SELECT 
            user_id,
            username,
            email,
            role,
            status,
            action_taken,
            risk_score,
            hour,
            minute,
            day_of_week,
            day_of_month,
            month,
            year,
            is_weekend,
            is_business_hours,
            ip_address,
            user_agent,
            device_fingerprint,
            browser,
            browser_version,
            os,
            os_version,
            device_type,
            screen_resolution,
            timezone,
            language,
            cookies_enabled,
            javascript_enabled,
            country,
            region,
            city,
            latitude,
            longitude,
            isp,
            connection_type,
            failed_attempts,
            is_known_device,
            is_known_location,
            location_mismatch,
            time_anomaly,
            velocity_check,
            session_id,
            previous_session_id
        FROM auth_logs
        WHERE risk_score IS NOT NULL 
          AND risk_score > 0
          AND risk_score <= 100
        ORDER BY created_at DESC
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        data = cursor.fetchall()
        
        print(f"   Found {len(data)} records with risk scores")
        return data
        
    except Error as e:
        print(f"❌ Error fetching data: {e}")
        return None
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def prepare_features(data):
    """Convert raw data to feature vectors - Using your actual columns"""
    X = []
    y = []
    
    for record in data:
        features = []
        
        # 1. Hour (0-23) normalized
        hour = record.get('hour', 12)
        if hour is None:
            hour = 12
        features.append(min(1.0, hour / 24))
        
        # 2. Minute normalized
        minute = record.get('minute', 0)
        if minute is None:
            minute = 0
        features.append(min(1.0, minute / 60))
        
        # 3. Day of week normalized
        day = record.get('day_of_week', 0)
        if day is None:
            day = 0
        features.append(day / 6)  # 0-6 -> 0-1
        
        # 4. Is weekend
        features.append(1 if record.get('is_weekend') else 0)
        
        # 5. Is business hours
        features.append(1 if record.get('is_business_hours') else 0)
        
        # 6. Failed attempts (capped)
        failed = record.get('failed_attempts', 0)
        if failed is None:
            failed = 0
        failed = min(int(failed), 10)
        features.append(failed / 10)
        
        # 7. Has device fingerprint
        features.append(1 if record.get('device_fingerprint') else 0)
        
        # 8. Browser reputation
        browser = record.get('browser', '')
        trusted = ['Chrome', 'Firefox', 'Safari', 'Edge']
        if browser and browser in trusted:
            features.append(1.0)
        elif browser:
            features.append(0.5)
        else:
            features.append(0)
        
        # 9. OS reputation
        os_name = record.get('os', '')
        trusted_os = ['Windows', 'macOS', 'iOS', 'Android']
        if os_name and os_name in trusted_os:
            features.append(1.0)
        elif os_name:
            features.append(0.5)
        else:
            features.append(0)
        
        # 10. Device type
        device_type = record.get('device_type', '')
        if device_type == 'mobile':
            features.append(1.0)
        elif device_type == 'tablet':
            features.append(0.7)
        elif device_type == 'desktop':
            features.append(0.3)
        else:
            features.append(0.5)
        
        # 11. Has country (location)
        features.append(1 if record.get('country') else 0)
        
        # 12. Location mismatch
        features.append(1 if record.get('location_mismatch') else 0)
        
        # 13. Is known device
        features.append(1 if record.get('is_known_device') else 0)
        
        # 14. Is known location
        features.append(1 if record.get('is_known_location') else 0)
        
        # 15. Time anomaly
        features.append(1 if record.get('time_anomaly') else 0)
        
        # 16. Velocity check (if exists)
        velocity = record.get('velocity_check', 0)
        if velocity is None:
            velocity = 0
        features.append(min(1.0, float(velocity) / 1000))
        
        # 17. Cookies enabled
        features.append(1 if record.get('cookies_enabled') else 0)
        
        # 18. JavaScript enabled
        features.append(1 if record.get('javascript_enabled') else 0)
        
        X.append(features)
        
        # Target: risk_score (normalized to 0-1)
        risk = record.get('risk_score', 50)
        if risk is None:
            risk = 50
        y.append(min(100, max(0, float(risk))) / 100)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_ensemble_models(X, y):
    """Train ensemble of models: Random Forest, Gradient Boosting, and Isolation Forest"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to store models
    models = {}
    
    # 1. Random Forest Regressor
    print("\n🌲 Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    models['random_forest'] = rf_model
    
    # 2. Gradient Boosting Regressor
    print("📈 Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    models['gradient_boosting'] = gb_model
    
    # 3. Isolation Forest (for anomaly detection)
    print("🔍 Training Isolation Forest...")
    isolation_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,  # Assume 10% anomalies
        random_state=42,
        n_jobs=-1
    )
    isolation_forest.fit(X_train_scaled)
    models['isolation_forest'] = isolation_forest
    
    # Evaluate regressors
    rf_train_score = rf_model.score(X_train_scaled, y_train)
    rf_test_score = rf_model.score(X_test_scaled, y_test)
    
    gb_train_score = gb_model.score(X_train_scaled, y_train)
    gb_test_score = gb_model.score(X_test_scaled, y_test)
    
    # Calculate prediction errors for ensemble
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    ensemble_mae = np.mean(np.abs(ensemble_pred - y_test)) * 100
    rf_mae = np.mean(np.abs(rf_pred - y_test)) * 100
    gb_mae = np.mean(np.abs(gb_pred - y_test)) * 100
    
    # Isolation Forest anomaly detection on test set
    if_pred = isolation_forest.predict(X_test_scaled)
    anomaly_rate = np.sum(if_pred == -1) / len(if_pred) * 100
    
    print(f"\n📊 Model Performance:")
    print(f"   Random Forest:")
    print(f"     - Training R²: {rf_train_score:.4f}")
    print(f"     - Testing R²:  {rf_test_score:.4f}")
    print(f"     - MAE: {rf_mae:.2f} points")
    print(f"   Gradient Boosting:")
    print(f"     - Training R²: {gb_train_score:.4f}")
    print(f"     - Testing R²:  {gb_test_score:.4f}")
    print(f"     - MAE: {gb_mae:.2f} points")
    print(f"   Ensemble (Average):")
    print(f"     - MAE: {ensemble_mae:.2f} points")
    print(f"   Isolation Forest:")
    print(f"     - Anomaly rate: {anomaly_rate:.1f}%")
    
    # Feature importance from Random Forest
    if hasattr(rf_model, 'feature_importances_'):
        print("\n🔑 Top 10 Important Features (Random Forest):")
        indices = np.argsort(rf_model.feature_importances_)[::-1][:10]
        feature_names = [
            'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'BusinessHours',
            'FailedAttempts', 'HasDevice', 'Browser', 'OS', 'DeviceType',
            'HasCountry', 'LocationMismatch', 'KnownDevice', 'KnownLocation',
            'TimeAnomaly', 'Velocity', 'Cookies', 'JavaScript'
        ]
        for i, idx in enumerate(indices):
            if idx < len(feature_names) and rf_model.feature_importances_[idx] > 0.01:
                print(f"   {i+1}. {feature_names[idx]}: {rf_model.feature_importances_[idx]:.4f}")
    
    return models, scaler, {
        'random_forest': {
            'train_r2': float(rf_train_score),
            'test_r2': float(rf_test_score),
            'mae': float(rf_mae)
        },
        'gradient_boosting': {
            'train_r2': float(gb_train_score),
            'test_r2': float(gb_test_score),
            'mae': float(gb_mae)
        },
        'ensemble_mae': float(ensemble_mae),
        'isolation_forest_anomaly_rate': float(anomaly_rate),
        'samples': len(X)
    }

# ============================================
# RISK ENGINE
# ============================================
class RiskEngine:
    def __init__(self):
        self.random_forest = None
        self.gradient_boosting = None
        self.isolation_forest = None
        self.scaler = None
        self.training_info = None
        self.load_model()
    
    def load_model(self):
        """Load models if exist"""
        rf_path = MODEL_PATH.replace('.pkl', '_rf.pkl')
        gb_path = MODEL_PATH.replace('.pkl', '_gb.pkl')
        
        if os.path.exists(rf_path) and os.path.exists(gb_path) and os.path.exists(SCALER_PATH):
            try:
                self.random_forest = joblib.load(rf_path)
                self.gradient_boosting = joblib.load(gb_path)
                self.scaler = joblib.load(SCALER_PATH)
                
                # Load Isolation Forest if exists
                if os.path.exists(ISOLATION_FOREST_PATH):
                    self.isolation_forest = joblib.load(ISOLATION_FOREST_PATH)
                
                # Load training info
                if os.path.exists(TRAINING_LOG):
                    with open(TRAINING_LOG, 'r') as f:
                        self.training_info = json.load(f)
                
                print("✅ Loaded ensemble models")
                if self.training_info:
                    print(f"   Trained on: {self.training_info.get('timestamp', 'Unknown')}")
                    print(f"   Samples: {self.training_info.get('samples', 0)}")
            except Exception as e:
                print(f"⚠️ Error loading models: {e}")
                self.random_forest = None
                self.gradient_boosting = None
        else:
            print("ℹ️ No existing models found - will use rule-based")
    
    def extract_features(self, data):
        """Extract features for prediction"""
        features = []
        
        # 1. Hour
        hour = int(data.get('hour', datetime.now().hour))
        features.append(min(1.0, hour / 24))
        
        # 2. Minute
        minute = int(data.get('minute', 0))
        features.append(min(1.0, minute / 60))
        
        # 3. Day of week
        day = int(data.get('day_of_week', datetime.now().weekday()))
        features.append(day / 6)
        
        # 4. Is weekend
        features.append(1 if data.get('is_weekend') else 0)
        
        # 5. Is business hours
        features.append(1 if data.get('is_business_hours') else 0)
        
        # 6. Failed attempts
        failed = min(int(data.get('failed_attempts', 0)), 10)
        features.append(failed / 10)
        
        # 7. Has device fingerprint
        features.append(1 if data.get('device_fingerprint') else 0)
        
        # 8. Browser
        browser = data.get('browser', '')
        trusted = ['Chrome', 'Firefox', 'Safari', 'Edge']
        if browser and browser in trusted:
            features.append(1.0)
        elif browser:
            features.append(0.5)
        else:
            features.append(0)
        
        # 9. OS
        os_name = data.get('os', '')
        trusted_os = ['Windows', 'macOS', 'iOS', 'Android']
        if os_name and os_name in trusted_os:
            features.append(1.0)
        elif os_name:
            features.append(0.5)
        else:
            features.append(0)
        
        # 10. Device type
        device_type = data.get('device_type', '')
        if device_type == 'mobile':
            features.append(1.0)
        elif device_type == 'tablet':
            features.append(0.7)
        elif device_type == 'desktop':
            features.append(0.3)
        else:
            features.append(0.5)
        
        # 11. Has country
        features.append(1 if data.get('country') else 0)
        
        # 12. Location mismatch
        features.append(1 if data.get('location_mismatch') else 0)
        
        # 13. Is known device
        features.append(1 if data.get('is_known_device') else 0)
        
        # 14. Is known location
        features.append(1 if data.get('is_known_location') else 0)
        
        # 15. Time anomaly
        features.append(1 if data.get('time_anomaly') else 0)
        
        # 16. Velocity
        velocity = float(data.get('velocity_check', 0))
        features.append(min(1.0, velocity / 1000))
        
        # 17. Cookies enabled
        features.append(1 if data.get('cookies_enabled') else 0)
        
        # 18. JavaScript enabled
        features.append(1 if data.get('javascript_enabled') else 0)
        
        return np.array(features, dtype=np.float32).reshape(1, -1)
    
    def predict(self, data):
        """Predict risk score using ensemble"""
        if self.random_forest and self.gradient_boosting and self.scaler:
            try:
                features = self.extract_features(data)
                features_scaled = self.scaler.transform(features)
                
                # Get predictions from both models
                rf_risk = self.random_forest.predict(features_scaled)[0]
                gb_risk = self.gradient_boosting.predict(features_scaled)[0]
                
                # Ensemble average
                risk_normalized = (rf_risk + gb_risk) / 2
                
                # Apply Isolation Forest anomaly detection if available
                if self.isolation_forest:
                    is_anomaly = self.isolation_forest.predict(features_scaled)[0]
                    if is_anomaly == -1:  # Anomaly detected
                        # Increase risk score by 20% if anomalous
                        risk_normalized = min(1.0, risk_normalized * 1.2)
                
                risk_score = float(risk_normalized) * 100
                risk_score = max(0, min(100, risk_score))
                
                method = 'ensemble'
                if self.isolation_forest:
                    method = 'ensemble+anomaly'
                
                return risk_score, method
                
            except Exception as e:
                print(f"ML prediction error: {e}")
        
        # Fallback to rule-based
        return self.rule_based(data), 'rule-based'
    
    def rule_based(self, data):
        """Rule-based risk calculation"""
        score = 50
        
        # Time
        hour = int(data.get('hour', datetime.now().hour))
        if hour < 6 or hour > 22:
            score += 20
        elif 9 <= hour <= 17:
            score -= 10
        
        # Weekend
        if data.get('is_weekend'):
            score += 10
        
        # Failed attempts
        failed = int(data.get('failed_attempts', 0))
        score += min(30, failed * 10)
        
        # Device
        if not data.get('device_fingerprint'):
            score += 25
        
        # Location
        if data.get('location_mismatch'):
            score += 30
        
        # Known device
        if not data.get('is_known_device'):
            score += 15
        
        return max(0, min(100, score))
    
    def get_info(self):
        """Get model information"""
        return {
            'model_loaded': self.random_forest is not None,
            'models': {
                'random_forest': self.random_forest is not None,
                'gradient_boosting': self.gradient_boosting is not None,
                'isolation_forest': self.isolation_forest is not None
            },
            'training_info': self.training_info
        }

# Initialize the engine
risk_engine = RiskEngine()

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/', methods=['GET'])
def home():
    """API home with status"""
    return jsonify({
        'name': 'MFA AI Risk Service',
        'status': 'running',
        'version': '3.2',
        'auto_train': True,
        'model': risk_engine.get_info(),
        'endpoints': {
            'GET /': 'This info',
            'GET /health': 'Health check',
            'POST /predict': 'Get risk score',
            'POST /retrain': 'Manually retrain model',
            'GET /model-info': 'Get model information'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': risk_engine.random_forest is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Get risk score for login attempt"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get prediction
        risk_score, method = risk_engine.predict(data)
        
        # Determine action
        if risk_score < 30:
            action = 'allow'
        elif risk_score < 70:
            action = 'challenge'
        else:
            action = 'block'
        
        # Generate request ID
        request_id = hashlib.md5(
            f"{data.get('user_id', 'unknown')}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:8]
        
        # Log
        print(f"\n[{datetime.now().isoformat()}] Request {request_id}")
        print(f"   User: {data.get('user_id')}")
        print(f"   Risk: {risk_score:.1f} ({action})")
        print(f"   Method: {method}")
        
        return jsonify({
            'success': True,
            'risk_score': round(risk_score, 1),
            'action': action,
            'method': method,
            'request_id': request_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'risk_score': 50,
            'action': 'challenge'
        }), 200

@app.route('/retrain', methods=['POST'])
def retrain():
    """Manually retrain the model"""
    print("\n📚 Manual retraining requested...")
    
    def train_thread():
        auto_train()
    
    thread = threading.Thread(target=train_thread)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Training started in background',
        'note': 'Check logs for progress'
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Get detailed model information"""
    info = risk_engine.get_info()
    
    if info['model_loaded'] and info['training_info']:
        return jsonify({
            'success': True,
            'model_loaded': True,
            'models': info['models'],
            'training_info': info['training_info']
        })
    else:
        return jsonify({
            'success': True,
            'model_loaded': False,
            'message': 'No trained model available - using rule-based'
        })

# ============================================
# MAIN - WITH AUTO-TRAINING ON STARTUP
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🔐 MFA AI RISK ASSESSMENT SERVICE v3.2")
    print("="*70)
    print(f"\n📡 Server Information:")
    print(f"   - Host: 0.0.0.0")
    print(f"   - Port: {PORT}")
    print(f"   - PID: {os.getpid()}")
    print(f"\n🤖 Auto-training: ENABLED")
    print(f"   - Using your actual database columns")
    print(f"   - Ensemble: Random Forest + Gradient Boosting + Isolation Forest")
    print(f"   - {len(fetch_training_data(1) or [])} records available")
    print(f"\n📡 Endpoints:")
    print(f"   - GET  http://localhost:{PORT}/")
    print(f"   - GET  http://localhost:{PORT}/health")
    print(f"   - POST http://localhost:{PORT}/predict")
    print(f"   - POST http://localhost:{PORT}/retrain")
    print(f"   - GET  http://localhost:{PORT}/model-info")
    
    # Run auto-training in a separate thread after server starts
    def start_auto_train():
        time.sleep(2)
        auto_train()
    
    train_thread = threading.Thread(target=start_auto_train)
    train_thread.daemon = True
    train_thread.start()
    
    print(f"\n⚡ Server is starting...")
    print("="*70)
    print("\nPress Ctrl+C to stop the server")
    print("-"*70)
    
    try:
        app.run(
            host='0.0.0.0', 
            port=PORT, 
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("\nPress Enter to exit...")