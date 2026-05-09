"""
MFA Model Analysis Script - COMPLETE VERSION with FULL Isolation Forest Integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
import os
import warnings
import sys
import traceback
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)

# Configuration
MODEL_PATH = 'models/risk_model_rf.pkl'
GB_MODEL_PATH = 'models/risk_model_gb.pkl'
ISOLATION_FOREST_PATH = 'models/isolation_forest.pkl'
SCALER_PATH = 'models/scaler.pkl'
TRAINING_LOG = 'models/training_log.json'

DB_CONFIG = {
    'host': 'localhost',
    'database': 'mfa_system',
    'user': 'root',
    'password': '',
    'port': 3306,
    'connection_timeout': 5,
    'use_pure': True,
    'autocommit': True
}

def log(msg):
    """Print with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()

def get_db_connection():
    """Establish connection to MySQL database"""
    try:
        connection = mysql.connector.connect(
            host='localhost',
            database='mfa_system',
            user='root',
            password='',
            port=3306,
            connection_timeout=5,
            use_pure=True,
            autocommit=True
        )
        return connection
    except Error as e:
        log(f"❌ Database connection error: {e}")
        return None

def fetch_test_data(limit=100):
    """Fetch test data from database with all required columns"""
    log(f"\n📊 Fetching up to {limit} records...")
    
    connection = get_db_connection()
    if not connection:
        log("❌ No database connection")
        return None, None
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check total records
        cursor.execute("SELECT COUNT(*) as total FROM auth_logs")
        result = cursor.fetchone()
        total = result['total'] if result else 0
        log(f"📊 Total records in auth_logs: {total}")
        
        if total == 0:
            log("❌ No records found")
            cursor.close()
            connection.close()
            return None, None
        
        # Full query with all columns
        query = """
        SELECT 
            id,
            user_id,
            risk_score,
            hour,
            minute,
            day_of_week,
            is_weekend,
            is_business_hours,
            failed_attempts,
            device_fingerprint,
            browser,
            os,
            device_type,
            country,
            location_mismatch,
            is_known_device,
            is_known_location,
            time_anomaly,
            velocity_check,
            cookies_enabled,
            javascript_enabled
        FROM auth_logs
        WHERE risk_score IS NOT NULL 
          AND risk_score > 0
          AND risk_score <= 100
        LIMIT %s
        """
        
        log("🔄 Running query...")
        cursor.execute(query, (limit,))
        data = cursor.fetchall()
        log(f"✅ Retrieved {len(data)} records")
        
        if not data:
            log("❌ No data returned")
            cursor.close()
            connection.close()
            return None, None
        
        # Show sample
        log(f"\n📝 Sample record (ID: {data[0]['id']}):")
        for key in ['risk_score', 'hour', 'minute', 'day_of_week', 'is_weekend', 'failed_attempts']:
            log(f"   {key}: {data[0].get(key)}")
        
        cursor.close()
        connection.close()
        
        # Prepare features
        log("\n🔧 Preparing features...")
        X, y = prepare_features(data)
        
        return X, y
        
    except Error as e:
        log(f"❌ Database error: {e}")
        traceback.print_exc()
        return None, None
    except Exception as e:
        log(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return None, None

def prepare_features(data):
    """Prepare all 18 features from raw data"""
    X = []
    y = []
    skipped = 0
    
    feature_names = [
        'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'BusinessHours',
        'FailedAttempts', 'HasDevice', 'Browser', 'OS', 'DeviceType',
        'HasCountry', 'LocationMismatch', 'KnownDevice', 'KnownLocation',
        'TimeAnomaly', 'Velocity', 'Cookies', 'JavaScript'
    ]
    
    log(f"🔧 Processing {len(data)} records for 18 features...")
    
    for i, record in enumerate(data):
        try:
            features = []
            
            # 1. Hour (0-23) normalized
            hour = record.get('hour', 12)
            if hour is None:
                hour = 12
            features.append(float(hour) / 24.0)
            
            # 2. Minute normalized
            minute = record.get('minute', 0)
            if minute is None:
                minute = 0
            features.append(float(minute) / 60.0)
            
            # 3. Day of week normalized
            day = record.get('day_of_week', 0)
            if day is None:
                day = 0
            features.append(float(day) / 6.0)
            
            # 4. Is weekend
            is_weekend = record.get('is_weekend', 0)
            features.append(1.0 if is_weekend and int(is_weekend) == 1 else 0.0)
            
            # 5. Is business hours
            is_business = record.get('is_business_hours', 0)
            features.append(1.0 if is_business and int(is_business) == 1 else 0.0)
            
            # 6. Failed attempts (capped at 10)
            failed = record.get('failed_attempts', 0)
            if failed is None:
                failed = 0
            failed = min(int(failed), 10)
            features.append(float(failed) / 10.0)
            
            # 7. Has device fingerprint
            device_fp = record.get('device_fingerprint')
            features.append(1.0 if device_fp and str(device_fp) != '' else 0.0)
            
            # 8. Browser reputation
            browser = record.get('browser', '')
            if browser and isinstance(browser, str):
                browser_lower = browser.lower()
                if any(x in browser_lower for x in ['chrome', 'firefox', 'safari', 'edge']):
                    features.append(1.0)
                else:
                    features.append(0.5)
            else:
                features.append(0.0)
            
            # 9. OS reputation
            os_name = record.get('os', '')
            if os_name and isinstance(os_name, str):
                os_lower = os_name.lower()
                if any(x in os_lower for x in ['windows', 'mac', 'ios', 'android', 'linux']):
                    features.append(1.0)
                else:
                    features.append(0.5)
            else:
                features.append(0.0)
            
            # 10. Device type
            device_type = record.get('device_type', '')
            if device_type and isinstance(device_type, str):
                dt_lower = device_type.lower()
                if 'mobile' in dt_lower:
                    features.append(1.0)
                elif 'tablet' in dt_lower:
                    features.append(0.7)
                elif any(x in dt_lower for x in ['desktop', 'pc', 'laptop']):
                    features.append(0.3)
                else:
                    features.append(0.5)
            else:
                features.append(0.5)
            
            # 11. Has country
            country = record.get('country')
            features.append(1.0 if country and str(country) != '' else 0.0)
            
            # 12. Location mismatch
            loc_mismatch = record.get('location_mismatch', 0)
            features.append(1.0 if loc_mismatch and int(loc_mismatch) == 1 else 0.0)
            
            # 13. Is known device
            known_device = record.get('is_known_device', 0)
            features.append(1.0 if known_device and int(known_device) == 1 else 0.0)
            
            # 14. Is known location
            known_loc = record.get('is_known_location', 0)
            features.append(1.0 if known_loc and int(known_loc) == 1 else 0.0)
            
            # 15. Time anomaly
            time_anomaly = record.get('time_anomaly', 0)
            features.append(1.0 if time_anomaly and int(time_anomaly) == 1 else 0.0)
            
            # 16. Velocity check
            velocity = record.get('velocity_check', 0)
            if velocity is None:
                velocity = 0
            features.append(min(1.0, float(velocity) / 1000.0))
            
            # 17. Cookies enabled
            cookies = record.get('cookies_enabled', 1)
            features.append(1.0 if cookies and int(cookies) == 1 else 0.0)
            
            # 18. JavaScript enabled
            js = record.get('javascript_enabled', 1)
            features.append(1.0 if js and int(js) == 1 else 0.0)
            
            X.append(features)
            
            # Target: risk_score normalized
            risk = record.get('risk_score', 50)
            if risk is None:
                risk = 50
            y.append(float(risk) / 100.0)
            
            # Show progress
            if (i + 1) % 20 == 0:
                log(f"   Processed {i + 1}/{len(data)} records...")
            
        except Exception as e:
            skipped += 1
            log(f"⚠️ Error processing record {i}: {e}")
            continue
    
    log(f"\n✅ Successfully processed {len(X)} records, skipped {skipped}")
    
    if len(X) == 0:
        log("❌ No valid records processed")
        return None, None
    
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    
    log(f"📊 Feature matrix shape: {X_array.shape} (should be 18 features)")
    log(f"📊 Risk score range: {y_array.min()*100:.1f} - {y_array.max()*100:.1f}")
    
    return X_array, y_array

def load_models():
    """Load trained models"""
    models = {}
    
    if os.path.exists(MODEL_PATH):
        log("📂 Loading Random Forest...")
        models['random_forest'] = joblib.load(MODEL_PATH)
        log("✅ Random Forest loaded")
    
    if os.path.exists(GB_MODEL_PATH):
        log("📂 Loading Gradient Boosting...")
        models['gradient_boosting'] = joblib.load(GB_MODEL_PATH)
        log("✅ Gradient Boosting loaded")
    
    if os.path.exists(SCALER_PATH):
        log("📂 Loading Scaler...")
        models['scaler'] = joblib.load(SCALER_PATH)
        log("✅ Scaler loaded")
    
    if os.path.exists(ISOLATION_FOREST_PATH):
        log("📂 Loading Isolation Forest...")
        models['isolation_forest'] = joblib.load(ISOLATION_FOREST_PATH)
        log("✅ Isolation Forest loaded")
    
    return models

def analyze_performance(models, X, y):
    """Comprehensive performance analysis including Isolation Forest metrics"""
    if X is None or len(X) == 0:
        log("❌ No data for performance analysis")
        return {}
    
    log("\n📈 Analyzing model performance...")
    
    if 'scaler' in models:
        log("🔄 Scaling features...")
        X_scaled = models['scaler'].transform(X)
    else:
        X_scaled = X
    
    y_true = y * 100
    results = {}
    
    # Random Forest
    if 'random_forest' in models:
        log("🌲 Random Forest predictions...")
        rf_pred = models['random_forest'].predict(X_scaled) * 100
        results['random_forest'] = {
            'mae': float(mean_absolute_error(y_true, rf_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, rf_pred))),
            'r2': float(r2_score(y_true, rf_pred)),
            'type': 'Supervised Regression'
        }
        log(f"   R²: {results['random_forest']['r2']:.3f}")
        log(f"   MAE: {results['random_forest']['mae']:.2f}")
    
    # Gradient Boosting
    if 'gradient_boosting' in models:
        log("📈 Gradient Boosting predictions...")
        gb_pred = models['gradient_boosting'].predict(X_scaled) * 100
        results['gradient_boosting'] = {
            'mae': float(mean_absolute_error(y_true, gb_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, gb_pred))),
            'r2': float(r2_score(y_true, gb_pred)),
            'type': 'Supervised Regression'
        }
        log(f"   R²: {results['gradient_boosting']['r2']:.3f}")
        log(f"   MAE: {results['gradient_boosting']['mae']:.2f}")
    
    # Ensemble
    if 'random_forest' in results and 'gradient_boosting' in results:
        log("🤝 Calculating ensemble...")
        rf_pred = models['random_forest'].predict(X_scaled) * 100
        gb_pred = models['gradient_boosting'].predict(X_scaled) * 100
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        results['ensemble'] = {
            'mae': float(mean_absolute_error(y_true, ensemble_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, ensemble_pred))),
            'r2': float(r2_score(y_true, ensemble_pred)),
            'type': 'Ensemble Regression'
        }
        log(f"   Ensemble R²: {results['ensemble']['r2']:.3f}")
        log(f"   Ensemble MAE: {results['ensemble']['mae']:.2f}")
    
    return results, rf_pred, gb_pred, ensemble_pred

def analyze_isolation_forest(models, X, y_true, save_path):
    """Analyze Isolation Forest anomaly detection performance"""
    if 'isolation_forest' not in models or 'scaler' not in models:
        log("⚠️ Isolation Forest or Scaler not available")
        return None
    
    log("\n🔍 Analyzing Isolation Forest anomaly detection...")
    
    # Scale features
    X_scaled = models['scaler'].transform(X)
    
    # Get predictions and scores
    if_pred = models['isolation_forest'].predict(X_scaled)
    if_scores = models['isolation_forest'].score_samples(X_scaled)
    
    # Calculate statistics
    anomaly_count = np.sum(if_pred == -1)
    normal_count = np.sum(if_pred == 1)
    anomaly_rate = anomaly_count / len(if_pred) * 100
    
    log(f"   Anomaly count: {anomaly_count} ({anomaly_rate:.1f}%)")
    log(f"   Normal count: {normal_count}")
    
    # Compare with risk categories
    y_true_cat = pd.cut(y_true, bins=[0, 30, 70, 100], labels=['Low', 'Medium', 'High'])
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        'actual_risk': y_true,
        'risk_category': y_true_cat,
        'is_anomaly': if_pred,
        'anomaly_score': if_scores
    })
    
    # Analyze anomaly distribution across risk categories
    anomaly_by_category = {}
    for cat in ['Low', 'Medium', 'High']:
        cat_mask = comparison['risk_category'] == cat
        cat_anomalies = comparison[cat_mask & (comparison['is_anomaly'] == -1)]
        cat_total = comparison[cat_mask].shape[0]
        
        if cat_total > 0:
            anomaly_by_category[cat] = {
                'total': int(cat_total),
                'anomalies': int(len(cat_anomalies)),
                'anomaly_rate': float(len(cat_anomalies) / cat_total * 100)
            }
            log(f"   {cat} Risk: {len(cat_anomalies)} anomalies out of {cat_total} ({len(cat_anomalies)/cat_total*100:.1f}%)")
    
    # Calculate precision/recall if we treat High Risk as "true anomalies"
    high_risk_mask = y_true >= 70
    if np.sum(high_risk_mask) > 0:
        # Treat High Risk as ground truth for anomalies
        tp = np.sum((if_pred == -1) & high_risk_mask)
        fp = np.sum((if_pred == -1) & ~high_risk_mask)
        fn = np.sum((if_pred == 1) & high_risk_mask)
        tn = np.sum((if_pred == 1) & ~high_risk_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        log(f"\n   Isolation Forest (using High Risk as ground truth):")
        log(f"     Accuracy: {accuracy:.3f}")
        log(f"     Precision: {precision:.3f}")
        log(f"     Recall: {recall:.3f}")
        log(f"     F1-Score: {f1:.3f}")
    else:
        precision = recall = f1 = accuracy = 0
    
    # Create Isolation Forest results dictionary
    isolation_results = {
        'anomaly_count': int(anomaly_count),
        'normal_count': int(normal_count),
        'anomaly_rate': float(anomaly_rate),
        'anomaly_by_category': anomaly_by_category,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'type': 'Unsupervised Anomaly Detection'
    }
    
    # Plot Isolation Forest results
    plot_isolation_forest_results(if_scores, if_pred, y_true, anomaly_by_category, save_path)
    
    return isolation_results

def plot_isolation_forest_results(if_scores, if_pred, y_true, anomaly_by_category, save_path):
    """Generate Isolation Forest visualizations"""
    log("\n🎨 Generating Isolation Forest plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Distribution of anomaly scores
    axes[0, 0].hist(if_scores, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(x=np.percentile(if_scores, 10), color='r', linestyle='--', 
                       linewidth=2, label='Anomaly Threshold (10th percentile)')
    axes[0, 0].set_xlabel('Anomaly Score', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Isolation Forest Anomaly Score Distribution', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Anomaly vs Normal counts
    anomaly_count = np.sum(if_pred == -1)
    normal_count = np.sum(if_pred == 1)
    
    bars = axes[0, 1].bar(['Normal (1)', 'Anomaly (-1)'], [normal_count, anomaly_count], 
                          color=['green', 'red'], edgecolor='black', alpha=0.7)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title(f'Anomaly Detection Results\n{anomaly_count} anomalies detected ({anomaly_count/len(if_pred)*100:.1f}%)', 
                        fontsize=14)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Anomaly scores vs Actual Risk
    colors = ['green' if p == 1 else 'red' for p in if_pred]
    scatter = axes[1, 0].scatter(y_true, if_scores, c=colors, alpha=0.6, 
                                 edgecolors='w', linewidth=0.5, s=60)
    axes[1, 0].axhline(y=np.percentile(if_scores, 10), color='r', linestyle='--', 
                       linewidth=2, label='Anomaly Threshold')
    axes[1, 0].set_xlabel('Actual Risk Score', fontsize=12)
    axes[1, 0].set_ylabel('Anomaly Score', fontsize=12)
    axes[1, 0].set_title('Anomaly Scores vs Actual Risk', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.6, label='Normal'),
                      Patch(facecolor='red', alpha=0.6, label='Anomaly')]
    axes[1, 0].legend(handles=legend_elements, loc='upper right')
    
    # 4. Anomaly rate by risk category
    categories = list(anomaly_by_category.keys())
    anomaly_rates = [anomaly_by_category[cat]['anomaly_rate'] for cat in categories]
    colors_by_cat = ['green', 'orange', 'red']
    
    bars = axes[1, 1].bar(categories, anomaly_rates, color=colors_by_cat, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Risk Category', fontsize=12)
    axes[1, 1].set_ylabel('Anomaly Rate (%)', fontsize=12)
    axes[1, 1].set_title('Anomaly Detection Rate by Risk Category', fontsize=14)
    
    # Add value labels on bars
    for bar, rate in zip(bars, anomaly_rates):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom', fontsize=11)
    
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, max(anomaly_rates) * 1.1])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/isolation_forest_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f"✅ Saved Isolation Forest analysis to {save_path}/isolation_forest_analysis.png")

def analyze_risk_categories(y_true, y_pred, thresholds=[30, 70]):
    """Analyze performance on risk categories (Low, Medium, High)"""
    log("\n📊 Analyzing risk categories...")
    
    # Convert to categories
    def to_category(score):
        if score < thresholds[0]:
            return 'Low'
        elif score < thresholds[1]:
            return 'Medium'
        else:
            return 'High'
    
    y_true_cat = [to_category(s) for s in y_true]
    y_pred_cat = [to_category(s) for s in y_pred]
    
    categories = ['Low', 'Medium', 'High']
    
    # Calculate metrics per category
    category_metrics = {}
    for cat in categories:
        mask = [t == cat for t in y_true_cat]
        if sum(mask) > 0:
            cat_true = [1 if t == cat else 0 for t in y_true_cat]
            cat_pred = [1 if p == cat else 0 for p in y_pred_cat]
            
            category_metrics[cat] = {
                'count': int(sum(mask)),
                'accuracy': float(accuracy_score(cat_true, cat_pred)),
                'precision': float(precision_score(cat_true, cat_pred, zero_division=0)),
                'recall': float(recall_score(cat_true, cat_pred, zero_division=0)),
                'f1': float(f1_score(cat_true, cat_pred, zero_division=0))
            }
    
    # Overall accuracy
    overall_acc = accuracy_score(y_true_cat, y_pred_cat)
    
    log(f"   Overall category accuracy: {overall_acc*100:.1f}%")
    for cat, metrics in category_metrics.items():
        log(f"   {cat}: F1={metrics['f1']:.3f}, Count={metrics['count']}")
    
    return {
        'overall_accuracy': float(overall_acc),
        'category_metrics': category_metrics,
        'confusion_matrix': confusion_matrix(y_true_cat, y_pred_cat, labels=categories).tolist()
    }

def analyze_feature_importance(models, feature_names, save_path):
    """Analyze and plot feature importance"""
    if 'random_forest' not in models:
        return None
    
    log("\n🔑 Analyzing feature importance...")
    
    importances = models['random_forest'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create feature importance dataframe
    feature_importance = []
    for i, idx in enumerate(indices):
        if idx < len(feature_names):
            feature_importance.append({
                'rank': i + 1,
                'feature': feature_names[idx],
                'importance': float(importances[idx]),
                'cumulative': float(np.sum(importances[indices[:i+1]]))
            })
    
    # Plot feature importance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top 15 features bar chart
    top_n = 15
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices if i < len(feature_names)]
    top_importances = [importances[i] for i in top_indices if i < len(feature_names)]
    
    bars = axes[0].barh(range(len(top_features)), top_importances[::-1], color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(len(top_features)))
    axes[0].set_yticklabels(top_features[::-1])
    axes[0].set_xlabel('Importance', fontsize=12)
    axes[0].set_title(f'Top {top_n} Feature Importances', fontsize=14)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_importances[::-1])):
        axes[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', va='center', fontsize=10)
    
    # Cumulative importance
    cumulative = np.cumsum(importances[indices])
    axes[1].plot(range(1, len(cumulative)+1), cumulative, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1].axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% threshold')
    axes[1].axhline(y=0.9, color='g', linestyle='--', linewidth=2, label='90% threshold')
    axes[1].set_xlabel('Number of Features', fontsize=12)
    axes[1].set_ylabel('Cumulative Importance', fontsize=12)
    axes[1].set_title('Cumulative Feature Importance', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, len(cumulative)+1])
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f"✅ Saved feature importance plot to {save_path}/feature_importance.png")
    
    return feature_importance

def plot_confusion_matrix(y_true, y_pred, save_path, model_name="Ensemble"):
    """Plot confusion matrix for risk categories"""
    log(f"\n📊 Generating confusion matrix for {model_name}...")
    
    # Convert to categories
    def to_category(score):
        if score < 30:
            return 'Low'
        elif score < 70:
            return 'Medium'
        else:
            return 'High'
    
    y_true_cat = [to_category(s) for s in y_true]
    y_pred_cat = [to_category(s) for s in y_pred]
    
    categories = ['Low', 'Medium', 'High']
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=categories)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Absolute numbers
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories, ax=axes[0],
                annot_kws={'size': 14})
    axes[0].set_xlabel('Predicted Risk Category', fontsize=12)
    axes[0].set_ylabel('Actual Risk Category', fontsize=12)
    axes[0].set_title(f'{model_name} - Confusion Matrix (Counts)', fontsize=14)
    
    # Percentages
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
                xticklabels=categories, yticklabels=categories, ax=axes[1],
                annot_kws={'size': 14})
    axes[1].set_xlabel('Predicted Risk Category', fontsize=12)
    axes[1].set_ylabel('Actual Risk Category', fontsize=12)
    axes[1].set_title(f'{model_name} - Confusion Matrix (%)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    log(f"✅ Saved confusion matrix to {save_path}/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")

def plot_model_comparison(performance, isolation_results, save_path):
    """Plot model comparison bar charts including Isolation Forest"""
    log("\n📊 Generating comprehensive model comparison plots...")
    
    # Prepare data for comparison
    models = []
    mae_values = []
    rmse_values = []
    r2_values = []
    
    # Add regression models
    for model_name, metrics in performance.items():
        if isinstance(metrics, dict) and 'r2' in metrics:
            models.append(model_name.replace('_', ' ').title())
            mae_values.append(metrics['mae'])
            rmse_values.append(metrics['rmse'])
            r2_values.append(metrics['r2'])
    
    # Create comparison figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MAE Comparison
    colors_mae = ['#2E86AB', '#A23B72', '#F18F01']
    bars1 = axes[0, 0].bar(models, mae_values, color=colors_mae[:len(models)], edgecolor='black', alpha=0.8)
    axes[0, 0].set_title('Mean Absolute Error (MAE) Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('MAE (lower is better)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, mae_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    # 2. RMSE Comparison
    bars2 = axes[0, 1].bar(models, rmse_values, color=colors_mae[:len(models)], edgecolor='black', alpha=0.8)
    axes[0, 1].set_title('Root Mean Square Error (RMSE) Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE (lower is better)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, rmse_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=11)
    
    # 3. R² Comparison
    bars3 = axes[1, 0].bar(models, r2_values, color=colors_mae[:len(models)], edgecolor='black', alpha=0.8)
    axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('R² (higher is better)', fontsize=12)
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars3, r2_values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    # 4. Isolation Forest Performance (if available)
    if isolation_results:
        if_metrics = ['Anomaly Rate', 'Precision', 'Recall', 'F1-Score']
        if_values = [
            isolation_results['anomaly_rate'],
            isolation_results['precision'] * 100,
            isolation_results['recall'] * 100,
            isolation_results['f1'] * 100
        ]
        
        bars4 = axes[1, 1].bar(if_metrics, if_values, color=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'], 
                              edgecolor='black', alpha=0.8)
        axes[1, 1].set_title('Isolation Forest Performance Metrics', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars4, if_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=11)
    else:
        axes[1, 1].text(0.5, 0.5, 'Isolation Forest\nNot Available', 
                       ha='center', va='center', fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Isolation Forest Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f"✅ Saved comprehensive model comparison to {save_path}/comprehensive_model_comparison.png")

def plot_actual_vs_predicted(models, X, y, save_path='analysis_plots'):
    """Generate enhanced actual vs predicted plots"""
    if X is None or len(X) == 0:
        return False
    
    os.makedirs(save_path, exist_ok=True)
    log("\n🎨 Generating enhanced plots...")
    
    if 'scaler' in models:
        X_scaled = models['scaler'].transform(X)
    else:
        X_scaled = X
    
    y_true = y * 100
    
    if 'random_forest' in models and 'gradient_boosting' in models:
        rf_pred = models['random_forest'].predict(X_scaled) * 100
        gb_pred = models['gradient_boosting'].predict(X_scaled) * 100
        ensemble_pred = (rf_pred + gb_pred) / 2
        
        y_pred_dict = {
            'Random Forest': rf_pred,
            'Gradient Boosting': gb_pred,
            'Ensemble': ensemble_pred
        }
        
        # Create enhanced plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
            row = idx // 2
            col = idx % 2
            
            # Scatter plot with perfect prediction line
            axes[row, col].scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='w', linewidth=0.5)
            axes[row, col].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Add confidence bands (±10)
            axes[row, col].fill_between([0, 100], [10, 110], [-10, 90], 
                                       alpha=0.1, color='green', label='±10% band')
            
            axes[row, col].set_xlabel('Actual Risk Score', fontsize=12)
            axes[row, col].set_ylabel('Predicted Risk Score', fontsize=12)
            axes[row, col].set_title(f'{model_name}\nR²={r2_score(y_true, y_pred):.3f}, MAE={mean_absolute_error(y_true, y_pred):.2f}', 
                                    fontsize=14)
            axes[row, col].legend(loc='lower right')
            axes[row, col].grid(True, alpha=0.3)
        
        # Error Distribution (4th subplot)
        errors = ensemble_pred - y_true
        axes[1, 1].hist(errors, bins=25, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[1, 1].axvline(x=np.mean(errors), color='blue', linestyle=':', linewidth=2, 
                          label=f'Mean: {np.mean(errors):.2f}')
        axes[1, 1].set_xlabel('Prediction Error', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title(f'Ensemble Error Distribution\nStd: {np.std(errors):.2f}', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/enhanced_analysis_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        log(f"✅ Saved enhanced plots to {save_path}/enhanced_analysis_plots.png")
        
        return y_pred_dict
    
    return None

def plot_error_analysis(y_true, y_pred_dict, save_path):
    """Detailed error analysis plots"""
    log("\n📊 Generating error analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for idx, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        row = idx // 2
        col = idx % 2
        
        errors = y_pred - y_true
        
        # Scatter plot of errors vs actual
        axes[row, col].scatter(y_true, errors, alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
        axes[row, col].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[row, col].axhline(y=10, color='orange', linestyle=':', linewidth=1)
        axes[row, col].axhline(y=-10, color='orange', linestyle=':', linewidth=1)
        axes[row, col].set_xlabel('Actual Risk Score', fontsize=11)
        axes[row, col].set_ylabel('Prediction Error', fontsize=11)
        axes[row, col].set_title(f'{model_name}\nMean Error: {np.mean(errors):.2f}', fontsize=12)
        axes[row, col].grid(True, alpha=0.3)
        
        # Add statistics box
        stats_text = f'Std: {np.std(errors):.2f}\nMAE: {np.mean(np.abs(errors)):.2f}'
        axes[row, col].text(0.05, 0.95, stats_text, transform=axes[row, col].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/error_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f"✅ Saved error analysis to {save_path}/error_analysis.png")

def generate_comprehensive_html_report(results, isolation_results, save_path):
    """Generate comprehensive HTML report with full Isolation Forest integration"""
    log("\n📄 Generating comprehensive HTML report...")
    
    # Find best performing model
    best_model = max(results['performance'].items(), key=lambda x: x[1]['r2'])[0]
    best_r2 = max([m['r2'] for m in results['performance'].values() if 'r2' in m])
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MFA ML Model Analysis Report - Complete with Isolation Forest</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #4CAF50; margin-top: 30px; border-left: 5px solid #4CAF50; padding-left: 15px; }}
            h3 {{ color: #555; margin-top: 25px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f8f8f8; }}
            tr:hover {{ background-color: #f0f0f0; }}
            .metric {{ font-weight: bold; color: #4CAF50; }}
            .plot {{ margin: 30px 0; text-align: center; background-color: #f9f9f9; padding: 20px; border-radius: 8px; }}
            .plot img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            .timestamp {{ color: #666; font-style: italic; margin-top: 20px; }}
            .summary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 10px; margin: 20px 0; }}
            .summary h2 {{ color: white; border-left: 5px solid white; }}
            .summary p {{ font-size: 16px; }}
            .badge {{ display: inline-block; padding: 5px 10px; border-radius: 20px; font-weight: bold; }}
            .badge-success {{ background-color: #4CAF50; color: white; }}
            .badge-warning {{ background-color: #ff9800; color: white; }}
            .badge-danger {{ background-color: #f44336; color: white; }}
            .badge-info {{ background-color: #2196F3; color: white; }}
            .model-card {{ display: inline-block; width: 200px; margin: 10px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MFA Machine Learning Model Analysis Report</h1>
            <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <table style="background-color: transparent; color: white; box-shadow: none;">
                    <tr>
                        <td><strong>Sample Size:</strong></td>
                        <td>{results['sample_size']} records</td>
                    </tr>
                    <tr>
                        <td><strong>Best Performing Model:</strong></td>
                        <td>{best_model.replace('_', ' ').title()} (R² = {best_r2:.3f})</td>
                    </tr>
                    <tr>
                        <td><strong>Isolation Forest:</strong></td>
                        <td>{isolation_results['anomaly_count']} anomalies detected ({isolation_results['anomaly_rate']:.1f}%)</td>
                    </tr>
                    <tr>
                        <td><strong>Overall Risk Category Accuracy:</strong></td>
                        <td>{results['risk_category_analysis']['overall_accuracy']*100:.1f}%</td>
                    </tr>
                </table>
            </div>
            
            <h2>Model Performance Metrics</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>R² Score</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                </tr>
    """
    
    for model_name, metrics in results['performance'].items():
        if isinstance(metrics, dict) and 'r2' in metrics:
            html_content += f"""
                <tr>
                    <td><strong>{model_name.replace('_', ' ').title()}</strong></td>
                    <td>{metrics.get('type', 'Supervised Regression')}</td>
                    <td>{metrics['r2']:.3f}</td>
                    <td>{metrics['mae']:.2f}</td>
                    <td>{metrics['rmse']:.2f}</td>
                </tr>
            """
    
    html_content += f"""
            </table>
            
            <h2>Isolation Forest Anomaly Detection</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td><strong>Anomaly Count</strong></td>
                    <td>{isolation_results['anomaly_count']}</td>
                </tr>
                <tr>
                    <td><strong>Normal Count</strong></td>
                    <td>{isolation_results['normal_count']}</td>
                </tr>
                <tr>
                    <td><strong>Anomaly Rate</strong></td>
                    <td>{isolation_results['anomaly_rate']:.1f}%</td>
                </tr>
                <tr>
                    <td><strong>Accuracy (vs High Risk)</strong></td>
                    <td>{isolation_results['accuracy']:.3f}</td>
                </tr>
                <tr>
                    <td><strong>Precision (vs High Risk)</strong></td>
                    <td>{isolation_results['precision']:.3f}</td>
                </tr>
                <tr>
                    <td><strong>Recall (vs High Risk)</strong></td>
                    <td>{isolation_results['recall']:.3f}</td>
                </tr>
                <tr>
                    <td><strong>F1-Score (vs High Risk)</strong></td>
                    <td>{isolation_results['f1']:.3f}</td>
                </tr>
            </table>
            
            <h3>Anomaly Distribution by Risk Category</h3>
            <table>
                <tr>
                    <th>Risk Category</th>
                    <th>Total Samples</th>
                    <th>Anomalies Detected</th>
                    <th>Anomaly Rate</th>
                </tr>
    """
    
    for cat, metrics in isolation_results['anomaly_by_category'].items():
        html_content += f"""
                <tr>
                    <td><strong>{cat}</strong></td>
                    <td>{metrics['total']}</td>
                    <td>{metrics['anomalies']}</td>
                    <td>{metrics['anomaly_rate']:.1f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Risk Category Analysis (Ensemble Model)</h2>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                </tr>
    """
    
    if 'risk_category_analysis' in results:
        for cat, metrics in results['risk_category_analysis']['category_metrics'].items():
            html_content += f"""
                <tr>
                    <td><strong>{cat}</strong></td>
                    <td>{metrics['count']}</td>
                    <td>{metrics['accuracy']*100:.1f}%</td>
                    <td>{metrics['precision']:.3f}</td>
                    <td>{metrics['recall']:.3f}</td>
                    <td>{metrics['f1']:.3f}</td>
                </tr>
            """
        html_content += f"""
            <tr>
                <td colspan="6"><strong>Overall Accuracy: {results['risk_category_analysis']['overall_accuracy']*100:.1f}%</strong></td>
            </tr>
        """
    
    html_content += """
            </table>
            
            <h2>Top 10 Feature Importances</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Feature</th>
                    <th>Importance</th>
                    <th>Cumulative</th>
                </tr>
    """
    
    if 'feature_importance' in results:
        for fi in results['feature_importance'][:10]:
            html_content += f"""
                <tr>
                    <td>{fi['rank']}</td>
                    <td>{fi['feature']}</td>
                    <td>{fi['importance']:.4f}</td>
                    <td>{fi['cumulative']:.4f}</td>
                </tr>
            """
    
    html_content += """
            </table>
            
            <h2>Visualizations</h2>
            <div class="plot">
                <h3>Enhanced Model Analysis (Actual vs Predicted)</h3>
                <img src="enhanced_analysis_plots.png" alt="Enhanced Analysis Plots">
            </div>
            <div class="plot">
                <h3>Isolation Forest Analysis</h3>
                <img src="isolation_forest_analysis.png" alt="Isolation Forest Analysis">
            </div>
            <div class="plot">
                <h3>Comprehensive Model Comparison (with Isolation Forest)</h3>
                <img src="comprehensive_model_comparison.png" alt="Comprehensive Model Comparison">
            </div>
            <div class="plot">
                <h3>Feature Importance</h3>
                <img src="feature_importance.png" alt="Feature Importance">
            </div>
            <div class="plot">
                <h3>Error Analysis</h3>
                <img src="error_analysis.png" alt="Error Analysis">
            </div>
            <div class="plot">
                <h3>Confusion Matrix - Ensemble</h3>
                <img src="confusion_matrix_ensemble.png" alt="Confusion Matrix Ensemble">
            </div>
            <div class="plot">
                <h3>Confusion Matrix - Random Forest</h3>
                <img src="confusion_matrix_random_forest.png" alt="Confusion Matrix Random Forest">
            </div>
            <div class="plot">
                <h3>Confusion Matrix - Gradient Boosting</h3>
                <img src="confusion_matrix_gradient_boosting.png" alt="Confusion Matrix Gradient Boosting">
            </div>
        </div>
    </body>
    </html>
    """
    
    report_path = f'{save_path}/comprehensive_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    log(f"✅ Saved comprehensive HTML report to {report_path}")

def main():
    """Main analysis function"""
    log("\n" + "="*80)
    log("📊 MFA MODEL ANALYSIS - COMPLETE VERSION WITH FULL ISOLATION FOREST INTEGRATION")
    log("="*80)
    
    # Load models
    log("\n📂 Loading models...")
    models = load_models()
    
    if not models:
        log("❌ No models found")
        return
    
    # Load training info
    if os.path.exists(TRAINING_LOG):
        with open(TRAINING_LOG, 'r') as f:
            training_info = json.load(f)
        log(f"\n📊 Training Info:")
        log(f"   - Samples: {training_info.get('samples', 0)}")
        log(f"   - Features: {training_info.get('features', 0)}")
    
    # Fetch test data
    log("\n📚 Fetching test data...")
    X, y = fetch_test_data(limit=100)
    
    if X is None or len(X) < 5:
        log("\n❌ Not enough test data")
        return
    
    log(f"\n✅ Loaded {len(X)} test samples")
    
    # Analyze performance
    log("\n" + "="*80)
    performance, rf_pred, gb_pred, ensemble_pred = analyze_performance(models, X, y)
    
    # Create timestamp for save path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'analysis_results_{timestamp}'
    os.makedirs(save_path, exist_ok=True)
    
    y_true = y * 100
    
    # Generate enhanced plots and get predictions
    y_pred_dict = plot_actual_vs_predicted(models, X, y, save_path)
    
    # Isolation Forest analysis
    isolation_results = analyze_isolation_forest(models, X, y_true, save_path)
    
    # Risk category analysis
    risk_category_results = analyze_risk_categories(y_true, ensemble_pred)
    
    # Feature importance analysis
    feature_names = [
        'Hour', 'Minute', 'DayOfWeek', 'IsWeekend', 'BusinessHours',
        'FailedAttempts', 'HasDevice', 'Browser', 'OS', 'DeviceType',
        'HasCountry', 'LocationMismatch', 'KnownDevice', 'KnownLocation',
        'TimeAnomaly', 'Velocity', 'Cookies', 'JavaScript'
    ]
    feature_importance = analyze_feature_importance(models, feature_names, save_path)
    
    # Confusion matrices
    plot_confusion_matrix(y_true, ensemble_pred, save_path, "Ensemble")
    plot_confusion_matrix(y_true, rf_pred, save_path, "Random Forest")
    plot_confusion_matrix(y_true, gb_pred, save_path, "Gradient Boosting")
    
    # Model comparison with Isolation Forest
    plot_model_comparison(performance, isolation_results, save_path)
    
    # Error analysis
    plot_error_analysis(y_true, y_pred_dict, save_path)
    
    # Compile all results
    results = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(X),
        'performance': performance,
        'risk_category_analysis': risk_category_results,
        'feature_importance': feature_importance
    }
    
    # Save results to JSON
    json_path = f'{save_path}/results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\n✅ Results saved to: {json_path}")
    
    # Generate comprehensive HTML report with Isolation Forest
    if isolation_results:
        generate_comprehensive_html_report(results, isolation_results, save_path)
    
    log(f"\n📁 All outputs saved in: {save_path}/")
    log("\n" + "="*80)
    log("✅ COMPLETE ANALYSIS FINISHED - ISOLATION FOREST FULLY INTEGRATED")
    log("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️ Interrupted")
    except Exception as e:
        log(f"\n❌ Error: {e}")
        traceback.print_exc()