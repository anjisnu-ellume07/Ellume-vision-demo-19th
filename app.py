import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from flask import session
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from wtforms import SubmitField, IntegerField, FloatField
from werkzeug.utils import secure_filename
import joblib
from datetime import datetime
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')
import requests
import uuid
import re
from markupsafe import Markup
import pickle
from datetime import datetime, timedelta
import time  # Added for sleep functionality
import tempfile
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from bson.json_util import dumps, loads
from dotenv import load_dotenv
import os
load_dotenv()
# MongoDB Configuration

MONGODB_URI = os.getenv("MONGODB_CONN_STRING")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Initialize MongoDB client
try:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Successfully connected to MongoDB")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    client = None
    collection = None
    fs = None

# PERSISTENT STORAGE CONFIGURATION
RESULTS_STORAGE_DIR = 'saved_results'
RESULTS_INDEX_FILE = os.path.join(RESULTS_STORAGE_DIR, 'results_index.json')

# Ensure storage directory exists
os.makedirs(RESULTS_STORAGE_DIR, exist_ok=True)

# In-memory store for analysis results (for current session)
TEMP_RESULT_CACHE = {}

# Persistent results index
PERSISTENT_RESULTS = {}

def push_to_mongodb(results_data, csv_filename=None):
    """Push analysis results to MongoDB, updating if path already exists"""
    if client is None or collection is None:
        print("MongoDB not available, skipping database push")
        return None
    
    try:
        # Prepare the document for MongoDB
        document = {
            'timestamp': datetime.now(),
            'filename': csv_filename or 'unknown',
            'total_samples': results_data.get('total_samples', 0),
            'anomaly_count': results_data.get('anomaly_count', 0),
            'anomaly_percentage': results_data.get('anomaly_percentage', 0),
            'device_id': results_data.get('device_id', 'unknown'),
            'date': results_data.get('date_yymmdd', 'unknown'),
            'plant_id': 'lPZ7JuDDGjzWmPATqJEu',
            'path': results_data.get('path', ''),
            'summary': {
                'daily_summaries': results_data.get('daily_summaries', {}),
                'severity_gauges': results_data.get('severity_gauges', {}),
                'gauge_errors': results_data.get('gauge_errors', [])
            },
            'metadata': {
                'contamination': results_data.get('contamination', 0.05),
                'n_explanations': results_data.get('n_explanations', 10)
            },
            # Store plot data directly in the document (no GridFS)
            'plot_data': results_data.get('plot_data', {})
        }
        
        # Check if a document with the same path already exists
        existing_doc = collection.find_one({'path': document['path']})
        
        if existing_doc:
            # Update the existing document
            result = collection.update_one(
                {'path': document['path']},
                {'$set': document}
            )
            document_id = str(existing_doc['_id'])
            print(f"Successfully updated existing document in MongoDB with ID: {document_id}")
        else:
            # Insert a new document
            result = collection.insert_one(document)
            document_id = str(result.inserted_id)
            print(f"Successfully inserted new document to MongoDB with ID: {document_id}")
        
        return document_id
        
    except Exception as e:
        print(f"Error pushing to MongoDB: {e}")
        return None

def get_from_mongodb(document_id):
    """Retrieve analysis results from MongoDB"""
    if client is None or collection is None:
        print("MongoDB not available")
        return None
    
    try:
        document = collection.find_one({'_id': ObjectId(document_id)})
        if document:
            # Convert ObjectId to string for JSON serialization
            document['_id'] = str(document['_id'])
            return document
        return None
    except Exception as e:
        print(f"Error retrieving from MongoDB: {e}")
        return None

def get_from_mongodb(document_id):
    """Retrieve analysis results from MongoDB"""
    # CORRECTED: Use 'is None' instead of 'not' for PyMongo objects
    if client is None or collection is None:
        print("MongoDB not available")
        return None
    
    try:
        document = collection.find_one({'_id': ObjectId(document_id)})
        if document:
            # Convert ObjectId to string for JSON serialization
            document['_id'] = str(document['_id'])
            
            # Retrieve plot data from GridFS if needed
            if 'plot_refs' in document:
                plot_data = {}
                for plot_name, file_id_str in document['plot_refs'].items():
                    try:
                        file_data = fs.get(ObjectId(file_id_str)).read()
                        plot_base64 = base64.b64encode(file_data).decode('utf-8')
                        plot_data[plot_name] = plot_base64
                    except Exception as e:
                        print(f"Error retrieving plot {plot_name} from GridFS: {e}")
                document['plot_data'] = plot_data
            
            return document
        return None
    except Exception as e:
        print(f"Error retrieving from MongoDB: {e}")
        return None
    
def load_results_index():
    """Load the persistent results index"""
    global PERSISTENT_RESULTS
    if os.path.exists(RESULTS_INDEX_FILE):
        try:
            with open(RESULTS_INDEX_FILE, 'r') as f:
                PERSISTENT_RESULTS = json.load(f)
        except Exception as e:
            print(f"Error loading results index: {e}")
            PERSISTENT_RESULTS = {}
    else:
        PERSISTENT_RESULTS = {}

def save_results_index():
    """Save the persistent results index"""
    try:
        with open(RESULTS_INDEX_FILE, 'w') as f:
            json.dump(PERSISTENT_RESULTS, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving results index: {e}")

def save_result_permanently(result_id, results_data):
    """Save analysis results permanently to disk"""
    try:
        result_file_path = os.path.join(RESULTS_STORAGE_DIR, f"{result_id}.pkl")
        
        # Save the full results data using pickle
        with open(result_file_path, 'wb') as f:
            pickle.dump(results_data, f)
        
        # Update the index with metadata
        PERSISTENT_RESULTS[result_id] = {
            'filename': results_data.get('filename', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'anomaly_count': results_data.get('anomaly_count', 0),
            'total_count': results_data.get('total_count', 0),
            'file_path': result_file_path
        }
        
        # Save the updated index
        save_results_index()
        
        print(f"Results saved permanently: {result_id}")
        return True
        
    except Exception as e:
        print(f"Error saving results permanently: {e}")
        return False

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, str):
        # Try to parse as JSON if it looks like JSON
        try:
            # Check if it looks like JSON (starts with { or [)
            if (obj.strip().startswith('{') and obj.strip().endswith('}')) or \
               (obj.strip().startswith('[') and obj.strip().endswith(']')):
                parsed = json.loads(obj)
                return convert_numpy_types(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
        return obj
    else:
        return obj

def load_result_from_disk(result_id):
    """Load analysis results from disk"""
    try:
        if result_id not in PERSISTENT_RESULTS:
            return None
            
        result_file_path = PERSISTENT_RESULTS[result_id]['file_path']
        
        if not os.path.exists(result_file_path):
            return None
            
        with open(result_file_path, 'rb') as f:
            results_data = pickle.load(f)
            
        return results_data
        
    except Exception as e:
        print(f"Error loading results from disk: {e}")
        return None

def cleanup_old_results(days_old=30):
    """Clean up results older than specified days"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days_old)
        results_to_remove = []
        
        for result_id, metadata in PERSISTENT_RESULTS.items():
            created_date = datetime.fromisoformat(metadata['created_at'])
            if created_date < cutoff_date:
                results_to_remove.append(result_id)
        
        for result_id in results_to_remove:
            result_file_path = PERSISTENT_RESULTS[result_id]['file_path']
            if os.path.exists(result_file_path):
                os.remove(result_file_path)
            del PERSISTENT_RESULTS[result_id]
        
        save_results_index()
        print(f"Cleaned up {len(results_to_remove)} old results")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")

# For LIME explanations
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Set random seed for reproducibility
np.random.seed(42)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 160 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load persistent results on startup
load_results_index()

# Hardcoded model path
MODEL_PATH = "solar_anomaly_detector.joblib"

# Forms
class UploadForm(FlaskForm):
    csv_file = FileField('CSV File', validators=[FileRequired()])
    n_explanations = IntegerField('Number of Explanations', default=10)
    contamination = FloatField('Contamination', default=0.05)
    submit = SubmitField('Analyze')


# Helper classes
class GroqAnalyzer:
    """Integration with Gemini API for detailed anomaly analysis"""
    
    def __init__(self, api_key):
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
        self.cache_file = "gemini_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load cached Gemini responses"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def _generate_cache_key(self, device_id, date_str, summary_data):
        """Generate a unique key for caching daily summaries"""
        return f"{device_id}|{date_str}|{hash(str(summary_data))}"
    
    def _format_groq_response(self, text):
        """Format Gemini response with proper HTML formatting"""
        # Convert markdown-like formatting to HTML
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
        
        # Convert numbered lists
        text = re.sub(r'(\d+)\.\s', r'<br><strong>\1.</strong> ', text)
        
        # Convert section headers
        text = re.sub(r'(\n|^)([A-Z\s]+):', r'<br><br><strong><u>\2:</u></strong><br>', text)
        
        # Convert bullet points into <ul><li>...</li></ul>
        def replace_bullets(match):
            items = match.group(0).strip().split('\n')
            list_items = ''.join(f'<li>{item[2:].strip()}</li>' for item in items if item.startswith('* '))
            return f'<ul>{list_items}</ul>'

        text = re.sub(r'((?:\* .+(?:\n|$))+)', replace_bullets, text)
        
        # Ensure proper line breaks
        text = text.replace('\n', '<br>')
        
        return Markup(text)
    
    def generate_daily_inverter_summary(self, device_id, date_str, anomaly_data, feature_analysis, state_info=""):
        """Generate a daily executive summary for a specific inverter"""
        
        # Prepare the prompt for Gemini - focused on daily summary
        prompt = f"""
**ROLE:** You are a seasoned Solar Performance Engineer with deep expertise in power electronics and inverter failure modes. You are analyzing data for a colleague, not an executive. Be technical, exploratory, and detailed.

**TASK:** Provide a deep technical analysis of the following inverter data. Focus on the "why" and "how," not just the "what." Avoid robotic, bullet-pointed conclusions. Write in a concise yet descriptive paragraph style.

**CRITICAL DIRECTIVE:** The "Root Cause Diagnosis" must be a detailed technical narrative. Do not isolate a single component with a confidence percentage. Instead, provide a brief overview of the most likely system (e.g., "An issue within the MPPT control loop...") and then dive into a detailed analysis of the 2-3 most probable specific component-level failures that could cause the observed patterns. Explain the failure mode for each candidate component and why the data supports it.

**DATA INPUT:**
- **Device ID:** {device_id}
- **Date:** {date_str}
- **Anomaly Trend:** {anomaly_data['trend']}
- **Peak Anomaly Time:** {anomaly_data['peak_time']}
- **Feature Analysis:** {feature_analysis}
{state_info}

**OUTPUT STRUCTURE:**

**1. Technical Performance Summary:**
Briefly state the overall operational status and the primary technical symptom (e.g., "The unit experienced significant power derating accompanied by erratic DC current imbalance...").

**2. Root Cause Analysis & Component-Level Investigation**
This is the main section. Write in depth anaysis in **very very short so that people read in bullets** and in less no of words that:
  - Starts with the most likely affected *system* (e.g., DC-side sensing, AC-side switching, control logic).
  - Details 2-3 potential root causes at the component level. For each candidate, describe:
      *   **The Component:** (e.g., "a degraded DC-link capacitor," "a failing current sensor on MPPT channel 1," "a damaged IGBT's gate driver")
      *   **Its Failure Mode:** (e.g., "increasing Equivalent Series Resistance (ESR) leading to voltage ripple")
      *   **Link to Data:** (e.g., "which would directly explain the observed high-frequency noise on the AC output and the drop in efficiency at peak power")

**3. Severity & Operational Impact:**
  - **Safety:** Comment on any risks (e.g., "Thermal runaway potential in the power module is elevated.")
  - **Production:** Estimate impact (e.g., "The ongoing losses are estimated at 15-20% due to constant clipping and fault shutdowns.")
  - **Urgency:** State the required response window based on failure progression.

**4. Recommended Investigation:** Prescribe the next technical step to confirm the diagnosis (e.g., "Perform on-site IV curve tracing on each string," "Analyze thermal imaging of the power cabinet," "Download event logs for ground fault errors.").
"""
       
        
        # Check cache first
        cache_key = self._generate_cache_key(device_id, date_str, feature_analysis)
        if cache_key in self.cache:
            print("Using cached Gemini response for daily summary")
            return self._format_groq_response(self.cache[cache_key])
        
        try:
            # Add a small delay to avoid timeout issues
            time.sleep(1.5)
            
            # Gemini API request format
            response = requests.post(
                f"{self.base_url}?key={self.api_key}",
                headers=self.headers,
                json={
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": f"You are an expert solar energy system diagnostician providing concise daily executive summaries for technical teams.\n\n{prompt}"
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 700
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()['candidates'][0]['content']['parts'][0]['text']
                # Cache the result
                self.cache[cache_key] = result
                self._save_cache()
                return self._format_groq_response(result)
            else:
                error_msg = f"Error calling Gemini API: {response.status_code} - {response.text}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error calling Gemini API: {str(e)}"
            print(error_msg)
            return error_msg

class SolarAnomalyDetector:
    def __init__(self, contamination=0.05, groq_api_key=None):
        """
        Solar Anomaly Detection System with Grok Integration
        """
        self.contamination = contamination
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.lime_explainer = None
        self.groq_analyzer = GroqAnalyzer(groq_api_key) if groq_api_key else None
        self.detailed_results = []  # Store all detailed analysis results
        self.daily_summaries = {}   # Store daily summaries by device
        
        # Load the hardcoded model
        self.load_model(MODEL_PATH)


    def load_and_preprocess_data(self, df):
        print("Loading and preprocessing data...")

        # Clean column names (remove spaces, make lowercase for consistency)
        df.columns = df.columns.str.strip().str.lower()

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Convert date to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Drop unwanted columns if present
        cols_to_remove = ['fault', 'count_data_reading_cycle']
        df = df.drop(columns=[c for c in cols_to_remove if c in df.columns], errors='ignore')
        
        # --- Power efficiency features ---
        df['efficiency'] = df['total_output_power'] / (df['total_dc_power'] + 1e-5)
        df['mppt_ratio'] = df['mppt1_dc_power'] / (df['mppt2_dc_power'] + 1e-5)
        
        # --- MPPT Current Imbalance ---
        df['mppt_current_imbalance'] = df[['mppt1_dc_current', 'mppt2_dc_current']].std(axis=1) / \
                                      df[['mppt1_dc_current', 'mppt2_dc_current']].mean(axis=1)

        # --- MPPT Voltage Imbalance ---
        df['mppt_voltage_imbalance'] = df[['mppt1_dc_voltage', 'mppt2_dc_voltage']].std(axis=1) / \
                                      df[['mppt1_dc_voltage', 'mppt2_dc_voltage']].mean(axis=1)

        # --- MPPT Efficiency ---
        df['mppt1_efficiency'] = df['mppt1_dc_power'] / (df['mppt1_dc_voltage'] * df['mppt1_dc_current'] + 1e-5)
        df['mppt2_efficiency'] = df['mppt2_dc_power'] / (df['mppt2_dc_voltage'] * df['mppt2_dc_current'] + 1e-5)
        df['mppt_eff_diff'] = abs(df['mppt1_efficiency'] - df['mppt2_efficiency'])

        # --- Voltage stability ---
        df['voltage_imbalance'] = df[['grid_voltage_v_1', 'grid_voltage_v_2', 'grid_voltage_v_3']].std(axis=1) / \
                                  df[['grid_voltage_v_1', 'grid_voltage_v_2', 'grid_voltage_v_3']].mean(axis=1)

        # --- Current stability ---
        df['current_imbalance'] = df[['grid_current_i_1', 'grid_current_i_2', 'grid_current_i_3']].std(axis=1) / \
                                  df[['grid_current_i_1', 'grid_current_i_2', 'grid_current_i_3']].mean(axis=1)

        # --- Temperature ---
        df['temp_diff'] = df['cell_temperature'] - df['cabinet_temperature']
        df['temp_ratio'] = df['cell_temperature'] / (df['avg_amb_temp'] + 1e-5)

        # --- Time-based features (needs 'hour' column present) ---
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        # --- Power degradation ---
        df['power_ratio'] = df['total_output_power'] / (df['apparent_power'] + 1e-5)

        # (Optional sanity check)
        print("Feature engineering complete. New shape:", df.shape)

        # Sort by timestamp if available
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Original dataset shape: {df.shape}")
        return df
    def check_state_4_analysis(self, df):
        """
        Analyze devices that are not in state 4 and determine duration of abnormal state
        """
        state_analysis = {}
        
        # Check if 'state' column exists
        if 'state' not in df.columns:
            return state_analysis
        
        # Check if 'q_serial' (device ID) and 'timestamp' columns exist
        if 'q_serial' not in df.columns or 'timestamp' not in df.columns:
            return state_analysis
        
        # Group by device
        for device_id, device_data in df.groupby('q_serial'):
            # Sort by timestamp
            device_data = device_data.sort_values('timestamp')
            
            # Find periods where device is not in state 4
            not_state_4 = device_data[device_data['state'] != 4]
            
            if len(not_state_4) > 0:
                # Calculate duration of abnormal state
                time_intervals = []
                current_interval_start = None
                prev_time = None
                
                for idx, row in not_state_4.iterrows():
                    current_time = row['timestamp']
                    
                    if current_interval_start is None:
                        current_interval_start = current_time
                    elif prev_time is not None:
                        # Check if gap is more than 30 minutes (2 intervals)
                        time_diff = (current_time - prev_time).total_seconds() / 60
                        if time_diff > 45:  # More than 45 minutes gap
                            # End current interval
                            duration = (prev_time - current_interval_start).total_seconds() / 3600  # Hours
                            time_intervals.append({
                                'start': current_interval_start,
                                'end': prev_time,
                                'duration_hours': duration
                            })
                            # Start new interval
                            current_interval_start = current_time
                    
                    prev_time = current_time
                
                # Add the last interval
                if current_interval_start is not None and prev_time is not None:
                    duration = (prev_time - current_interval_start).total_seconds() / 3600  # Hours
                    time_intervals.append({
                        'start': current_interval_start,
                        'end': prev_time,
                        'duration_hours': duration
                    })
                
                # Calculate statistics
                total_abnormal_hours = sum(interval['duration_hours'] for interval in time_intervals)
                max_duration = max(interval['duration_hours'] for interval in time_intervals) if time_intervals else 0
                avg_duration = total_abnormal_hours / len(time_intervals) if time_intervals else 0
                
                # Check if device was never in state 4 throughout the day
                day_start = device_data['timestamp'].min().replace(hour=0, minute=0, second=0, microsecond=0)
                day_end = day_start + pd.Timedelta(days=1)
                day_data = device_data[(device_data['timestamp'] >= day_start) & (device_data['timestamp'] < day_end)]
                
                never_state_4 = day_data[day_data['state'] == 4].empty if len(day_data) > 0 else False
                
                state_analysis[device_id] = {
                    'abnormal_intervals': time_intervals,
                    'total_abnormal_hours': total_abnormal_hours,
                    'max_duration_hours': max_duration,
                    'avg_duration_hours': avg_duration,
                    'never_in_state_4': never_state_4,
                    'abnormal_percentage': (len(not_state_4) / len(device_data)) * 100
                }
        
        return state_analysis
    def feature_selection_and_engineering(self, df):
        """Intelligent feature selection and engineering for solar anomaly detection"""
        print("Performing feature selection and engineering...")

        # Remove columns that are not useful for anomaly detection
        columns_to_remove = [
            'timestamp',  # Will use engineered time features instead
            'count_data_reading_cycle',  # Just a counter
            'date',       # Redundant with timestamp
            'today_e_increment'  # Derived feature, might cause data leakage
        ]

        # Keep only relevant columns
        df_clean = df.drop(columns=columns_to_remove, errors='ignore')

        # Feature Engineering
        print("Engineering new features...")

        # Time-based features (from timestamp)
        if 'hour' in df.columns:
            df_clean['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df_clean['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df_clean['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df_clean['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df_clean['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df_clean['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Power efficiency ratios
        df_clean['dc_to_ac_efficiency'] = df_clean['total_output_power'] / (df_clean['total_dc_power'] + 1e-6)
        df_clean['mppt1_efficiency'] = df_clean['mppt1_dc_power'] / (df_clean['mppt1_dc_voltage'] * df_clean['mppt1_dc_current'] + 1e-6)
        df_clean['mppt2_efficiency'] = df_clean['mppt2_dc_power'] / (df_clean['mppt2_dc_voltage'] * df_clean['mppt2_dc_current'] + 1e-6)

        # Power balance features
        df_clean['mppt_power_balance'] = abs(df_clean['mppt1_dc_power'] - df_clean['mppt2_dc_power'])
        df_clean['total_mppt_power'] = df_clean['mppt1_dc_power'] + df_clean['mppt2_dc_power']
        df_clean['mppt_dc_power_diff'] = abs(df_clean['total_mppt_power'] - df_clean['total_dc_power'])

        # Voltage and current ratios
        df_clean['voltage_imbalance_12'] = abs(df_clean['grid_voltage_v_1'] - df_clean['grid_voltage_v_2'])
        df_clean['voltage_imbalance_13'] = abs(df_clean['grid_voltage_v_1'] - df_clean['grid_voltage_v_3'])
        df_clean['voltage_imbalance_23'] = abs(df_clean['grid_voltage_v_2'] - df_clean['grid_voltage_v_3'])

        df_clean['current_imbalance_12'] = abs(df_clean['grid_current_i_1'] - df_clean['grid_current_i_2'])
        df_clean['current_imbalance_13'] = abs(df_clean['grid_current_i_1'] - df_clean['grid_current_i_3'])
        df_clean['current_imbalance_23'] = abs(df_clean['grid_current_i_2'] - df_clean['grid_current_i_3'])

        # Temperature-based features
        df_clean['temp_difference'] = df_clean['max_amb_temp'] - df_clean['min_amb_temp']
        df_clean['cabinet_temp_deviation'] = df_clean['cabinet_temperature'] - df_clean['avg_amb_temp']
        df_clean['cell_temp_deviation'] = df_clean['cell_temperature'] - df_clean['avg_amb_temp']

        # Power factor and reactive power features
        df_clean['power_factor'] = df_clean['total_output_power'] / (df_clean['apparent_power'] + 1e-6)
        df_clean['reactive_power_ratio'] = df_clean['reactive_power'] / (df_clean['total_output_power'] + 1e-6)

        # Irradiance-based features
        df_clean['power_per_irradiance'] = df_clean['total_output_power'] / (df_clean['avg_irrad'] + 1e-6)

        # Remove original time columns as we have engineered features
        time_cols_to_remove = ['hour', 'day_of_week', 'month']
        df_clean = df_clean.drop(columns=time_cols_to_remove, errors='ignore')

        # Handle infinite and NaN values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        df_clean = df_clean.fillna(df_clean.median())

        print(f"After feature engineering shape: {df_clean.shape}")
        print(f"Features selected: {list(df_clean.columns)}")

        return df_clean

    def load_model(self, model_path):
        """Load a trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.contamination = model_data['contamination']

        print(f"Model loaded from {model_path}")
        return self

    def predict_anomalies(self, X_test):
        """Predict anomalies on test data"""
        print("Predicting anomalies...")

        # Scale test data
        X_test_scaled = self.scaler.transform(X_test)

        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_test_scaled)

        # Get anomaly scores
        anomaly_scores = self.model.decision_function(X_test_scaled)

        # Convert predictions to binary (1 for anomaly, 0 for normal)
        binary_predictions = np.where(predictions == -1, 1, 0)

        return binary_predictions, anomaly_scores, X_test_scaled

    def explain_anomalies_with_lime_and_groq(self, X_test, X_test_scaled, anomaly_indices, original_df, n_explanations=10):
        """Generate LIME explanations and get detailed Grok analysis for detected anomalies"""
        if not LIME_AVAILABLE or self.lime_explainer is None:
            print("LIME not available for explanations")
            return {}  # Return empty dict instead of None

        print(f"\n" + "="*70)
        print("DAILY INVERTER SUMMARY ANALYSIS")
        print("="*70)

        # Add state analysis
        state_analysis = self.check_state_4_analysis(original_df)
        
        # Group anomalies by device and date
        device_anomalies = {}
        
        for idx in anomaly_indices:
            device_id = original_df.loc[idx, 'q_serial'] if 'q_serial' in original_df.columns else f"Device_{idx}"
            timestamp = original_df.loc[idx, 'timestamp'] if 'timestamp' in original_df.columns else "Unknown"
            
            # Extract date for grouping
            if hasattr(timestamp, 'date'):
                date_str = timestamp.date().isoformat()
            else:
                date_str = "unknown_date"
            
            # Initialize device entry if not exists
            if device_id not in device_anomalies:
                device_anomalies[device_id] = {}
            
            # Initialize date entry if not exists
            if date_str not in device_anomalies[device_id]:
                device_anomalies[device_id][date_str] = {
                    'indices': [],
                    'timestamps': [],
                    'scores': []
                }
            
            # Add anomaly to device/date group
            device_anomalies[device_id][date_str]['indices'].append(idx)
            device_anomalies[device_id][date_str]['timestamps'].append(timestamp)
            device_anomalies[device_id][date_str]['scores'].append(
                self.model.decision_function(X_test_scaled[idx].reshape(1, -1))[0]
            )

        # Initialize daily_summaries as a dictionary
        daily_summaries = {}

        # Generate daily summaries for each device
        for device_id, dates in device_anomalies.items():
            # Initialize device entry in daily_summaries
            if device_id not in daily_summaries:
                daily_summaries[device_id] = {}
                
            for date_str, date_data in dates.items():
                print(f"\n" + "="*70)
                print(f"DAILY SUMMARY: {device_id} - {date_str}")
                print("="*70)
                print(f"Total anomalies: {len(date_data['indices'])}")
                
                # Calculate daily statistics
                anomaly_times = [ts.hour for ts in date_data['timestamps'] if hasattr(ts, 'hour')]
                peak_time = max(set(anomaly_times), key=anomaly_times.count) if anomaly_times else "Unknown"
                
                # Determine trend (comparing to previous days if available)
                trend = "Unknown"
                if len(date_data['scores']) > 1:
                    avg_score = np.mean(date_data['scores'])
                    trend = "Increasing" if avg_score > np.median(date_data['scores']) else "Stable"
                
                # Get feature analysis for the day (average of top features across anomalies)
                day_features = {}
                for idx in date_data['indices']:
                    try:
                        # Create a wrapper function for LIME
                        def predict_fn(X):
                            preds = self.model.predict(X)
                            probs = np.zeros((len(preds), 2))
                            probs[:, 0] = (preds == 1).astype(float)  # Normal
                            probs[:, 1] = (preds == -1).astype(float)  # Anomaly
                            return probs

                        # Generate LIME explanation
                        explanation = self.lime_explainer.explain_instance(
                            X_test_scaled[idx].reshape(1, -1)[0],
                            predict_fn,
                            num_features=10,
                            top_labels=1
                        )

                        # Get LIME feature explanations
                        lime_features = explanation.as_list(label=1)
                        
                        for feature, weight in lime_features:
                            if feature not in day_features:
                                day_features[feature] = []
                            day_features[feature].append(weight)
                    
                    except Exception as e:
                        print(f"Error generating explanation for index {idx}: {str(e)}")
                        continue
                
                # Calculate average weights for each feature
                avg_features = []
                for feature, weights in day_features.items():
                    avg_weight = np.mean(weights)
                    avg_features.append((feature, avg_weight))
                
                # Sort by absolute weight
                avg_features.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Format feature analysis
                feature_analysis = "\n".join([f"- {feature}: Avg Impact = {weight:.4f}" 
                                            for feature, weight in avg_features[:5]])  # Top 5 features only
                
                print(f"\nTOP FEATURES FOR DAY:")
                print("-" * 40)
                for feature, weight in avg_features[:5]:
                    print(f"  {feature:<30}: {weight:>8.4f}")

                # Prepare anomaly data for Grok
                anomaly_data = {
                    'total_anomalies': len(date_data['indices']),
                    'trend': trend,
                    'peak_time': f"{peak_time}:00" if peak_time != "Unknown" else "Unknown"
                }
                
                # Add state information to the prompt if device has state issues
                state_info = ""
                if device_id in state_analysis:
                    state_data = state_analysis[device_id]
                    state_info = f"""
    **STATE ANALYSIS:**
    - Device was not in normal operating state (state 4) for {state_data['total_abnormal_hours']:.2f} hours
    - Longest continuous abnormal period: {state_data['max_duration_hours']:.2f} hours
    - Percentage of time in abnormal state: {state_data['abnormal_percentage']:.1f}%
    - Never in normal state throughout the day: {'Yes' if state_data['never_in_state_4'] else 'No'}
    """
                
                # Get Grok daily summary if available
                groq_summary = "Grok analysis not available (API key not provided)"
                if self.groq_analyzer:
                    print(f"\n GROK DAILY EXECUTIVE SUMMARY:")
                    print("-" * 40)
                    
                    try:
                        # Update prompt to include state information
                        prompt = f"""
    **ROLE:** You are a seasoned Solar Performance Engineer with deep expertise in power electronics and inverter failure modes. You are analyzing data for a colleague, not an executive. Be technical, exploratory, and detailed.

    **TASK:** Provide a deep technical analysis of the following inverter data. Focus on the "why" and "how," not just the "what." Avoid robotic, bullet-pointed conclusions. Write in a concise yet descriptive paragraph style.

    **CRITICAL DIRECTIVE:** The "Root Cause Diagnosis" must be a detailed technical narrative. Do not isolate a single component with a confidence percentage. Instead, provide a brief overview of the most likely system (e.g., "An issue within the MPPT control loop...") and then dive into a detailed analysis of the 2-3 most probable specific component-level failures that could cause the observed patterns. Explain the failure mode for each candidate component and why the data supports it.

    **DATA INPUT:**
    - **Device ID:** {device_id}
    - **Date:** {date_str}
    - **Anomaly Trend:** {anomaly_data['trend']}
    - **Peak Anomaly Time:** {anomaly_data['peak_time']}
    - **Feature Analysis:** {feature_analysis}
    {state_info}

    **OUTPUT STRUCTURE:**

    **1. Technical Performance Summary:**
    Briefly state the overall operational status and the primary technical symptom (e.g., "The unit experienced significant power derating accompanied by erratic DC current imbalance...").

    **2.Root Cause Analysis & Component-Level Investigation**
    This is the main section. Write in depth anaysis in very very short so that people read in bullets and in less no of words that:
    - Starts with the most likely affected *system* (e.g., DC-side sensing, AC-side switching, control logic).
    - Details 2-3 potential root causes at the component level. For each candidate, describe:
        *   **The Component:** (e.g., "a degraded DC-link capacitor," "a failing current sensor on MPPT channel 1," "a damaged IGBT's gate driver")
        *   **Its Failure Mode:** (e.g., "increasing Equivalent Series Resistance (ESR) leading to voltage ripple")
        *   **Link to Data:** (e.g., "which would directly explain the observed high-frequency noise on the AC output and the drop in efficiency at peak power")

    **3. Severity & Operational Impact:**
    - **Safety:** Comment on any risks (e.g., "Thermal runaway potential in the power module is elevated.")
    - **Production:** Estimate impact (e.g., "The ongoing losses are estimated at 15-20% due to constant clipping and fault shutdowns.")
    - **Urgency:** State the required response window based on failure progression.

    **4. Recommended Investigation:** Prescribe the next technical step to confirm the diagnosis (e.g., "Perform on-site IV curve tracing on each string," "Analyze thermal imaging of the power cabinet," "Download event logs for ground fault errors.").
    """
                        
                        groq_summary = self.groq_analyzer.generate_daily_inverter_summary(
                            device_id, 
                            date_str, 
                            anomaly_data, 
                            feature_analysis,
                            state_info
                        )
                        print(groq_summary)
                    except Exception as e:
                        print(f"Error getting Grok summary: {str(e)}")
                        groq_summary = f"Grok analysis failed: {str(e)}"
                else:
                    print("\nGrok analysis not available (API key not provided)")

                # Store daily summary with state information
                daily_summaries[device_id][date_str] = {
                    'summary': groq_summary,
                    'anomaly_count': int(len(date_data['indices'])),
                    'avg_severity': float(np.mean(date_data['scores'])*10),
                    'peak_time': peak_time,
                    'trend': trend,
                    'state_analysis': state_analysis.get(device_id, {})
                }

        # Ensure we return the dictionary we've been building
        return daily_summaries
    
    def create_simple_gauge_fallback(self, severity_score, device_id, date_str):
        """Create a simple bar-style gauge as fallback"""
        try:
            fig, ax = plt.subplots(figsize=(6, 2))
            
            # Normalize score
            normalized_score = min(max(abs(severity_score) * 50, 0), 100)
            
            # Create horizontal bar
            bar_width = normalized_score
            colors = ['green' if normalized_score < 25 else 
                    'yellow' if normalized_score < 50 else 
                    'orange' if normalized_score < 75 else 'red']
            
            ax.barh(0, bar_width, height=0.5, color=colors[0], alpha=0.8)
            ax.barh(0, 100, height=0.5, color='lightgray', alpha=0.3)
            
            # Add text
            ax.text(50, 0, f"{normalized_score:.1f}", ha='center', va='center', 
                fontweight='bold', fontsize=14)
            
            # Set limits and remove axes
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            plt.title(f"Severity: {device_id} - {date_str}", fontsize=10)
            plt.tight_layout()
            
            # Save to buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return plot_data
            
        except Exception as e:
            print(f"Error creating fallback gauge: {str(e)}")
            return None
        
    def create_severity_gauge(self, severity_score, device_id, date_str):
        """Create a speedometer-style severity gauge with proper score normalization"""
        try:
            # Create figure without polar projection first, then add it
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection='polar')
            
            # FIXED: Proper severity score calculation
            # Calculate normalized score based on anomaly severity and production impact
            
            # Method 1: If you have production loss data directly
            # Check if we can determine production loss from the data
            # This would need to be calculated from your actual data
            # For now, we'll use a better normalization of the anomaly score
            
            # Convert anomaly score to a meaningful severity percentage
            # Anomaly scores are typically between -2 and 2 for Isolation Forest
            # Negative values indicate anomalies, positive values indicate normal behavior
            
            if severity_score <= -1.5:  # Very severe anomaly
                normalized_score = 85 + (abs(severity_score + 1.5) * 30)  # 85-100%
            elif severity_score <= -1.0:  # Severe anomaly  
                normalized_score = 60 + (abs(severity_score + 1.0) * 50)  # 60-85%
            elif severity_score <= -0.5:  # Moderate anomaly
                normalized_score = 35 + (abs(severity_score + 0.5) * 50)  # 35-60%
            elif severity_score <= 0:     # Minor anomaly
                normalized_score = 10 + (abs(severity_score) * 50)       # 10-35%
            else:                         # Normal behavior (should rarely happen in anomaly context)
                normalized_score = max(0, 10 - (severity_score * 10))    # 0-10%
            
            # Clamp the score to 0-100 range
            normalized_score = min(max(normalized_score, 0), 100)
            
            # TODO: Add production loss calculation here when available
            # If you have actual production loss data, use this logic:
            """
            # Example production loss calculation (uncomment and modify as needed):
            if hasattr(self, 'production_loss_data'):  # If you have production loss data
                # Get production loss for this device and date
                production_loss = self.get_production_loss(device_id, date_str)
                if production_loss >= 95:  # 95%+ production loss = 100% severity
                    normalized_score = 100
                elif production_loss >= 80:  # 80-95% loss = 80-99% severity
                    normalized_score = 80 + (production_loss - 80) * 1.27
                elif production_loss >= 50:  # 50-80% loss = 50-79% severity  
                    normalized_score = 50 + (production_loss - 50)
                # Use production loss directly if it's more severe than anomaly score
                normalized_score = max(normalized_score, production_loss)
            """
            
            # Define gauge parameters
            angle_range = np.linspace(0, np.pi, 100)  # Half circle (0 to π)
            needle_angle = normalized_score / 100 * np.pi
            
            # Create gauge background (light gray arc)
            radius = 1.0
            ax.plot(angle_range, np.ones(100) * radius, 
                color='lightgray', linewidth=15, alpha=0.3, solid_capstyle='round')
            
            # Create colored segments with better color mapping
            segment_size = len(angle_range) // 4
            colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']  # Green, Yellow, Orange, Red
            
            for i, color in enumerate(colors):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < 3 else len(angle_range)
                segment_angles = angle_range[start_idx:end_idx]
                
                if len(segment_angles) > 0:
                    ax.plot(segment_angles, np.ones(len(segment_angles)) * radius,
                        color=color, linewidth=15, alpha=0.8, solid_capstyle='round')
            
            # Add needle (black line from center to current position)
            ax.plot([needle_angle, needle_angle], [0, 0.8], 
                color='black', linewidth=4, solid_capstyle='round')
            
            # Add needle tip (circle)
            ax.scatter(needle_angle, 0.8, color='black', s=50, zorder=5)
            
            # Add center dot
            ax.scatter(0, 0, color='black', s=100, zorder=5)
            
            # Add score text in center with integer display
            ax.text(np.pi/2, 0.3, f"{int(normalized_score)}%", 
                ha='center', va='center', fontsize=16, fontweight='bold',
                transform=ax.transData)
            
            ax.text(np.pi/2, 0.1, "Severity Score", 
                ha='center', va='center', fontsize=10,
                transform=ax.transData)
            
            # Add severity level text based on score
            if normalized_score >= 85:
                severity_level = "CRITICAL"
                level_color = '#DC143C'
            elif normalized_score >= 60:
                severity_level = "HIGH"  
                level_color = '#FF8C00'
            elif normalized_score >= 35:
                severity_level = "MEDIUM"
                level_color = '#FFD700'
            else:
                severity_level = "LOW"
                level_color = '#2E8B57'
                
            ax.text(np.pi/2, -0.1, severity_level, 
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=level_color, transform=ax.transData)
            
            # Set title
            plt.title(f"Severity: {device_id}\n{date_str}", pad=20, fontsize=12)
            
            # Configure the polar plot
            ax.set_ylim(0, 1.2)
            ax.set_xlim(0, np.pi)
            
            # Remove all labels and ticks
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            
            plt.tight_layout()
            
            # Save plot to bytes buffer
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            buf.seek(0)
            
            # Encode to base64
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)  # Important: close the figure to prevent memory leaks
            
            return plot_data
            
        except Exception as e:
            print(f"Error creating severity gauge: {str(e)}")
            # Return a simple fallback gauge
            return self.create_simple_gauge_fallback(severity_score, device_id, date_str)

    def get_production_loss(self, device_id, date_str):
        """
        Calculate production loss percentage for a given device and date
        This method should be implemented based on your actual data structure
        """
        # TODO: Implement this method based on your data
        # This is a placeholder - you'll need to implement the actual logic
        # based on how you calculate production loss in your system
        
        # Example implementation (modify as needed):
        """
        try:
            # Get expected production for this device/date
            expected_production = self.get_expected_production(device_id, date_str)
            
            # Get actual production for this device/date  
            actual_production = self.get_actual_production(device_id, date_str)
            
            # Calculate loss percentage
            if expected_production > 0:
                production_loss = ((expected_production - actual_production) / expected_production) * 100
                return max(0, min(100, production_loss))  # Clamp to 0-100%
            
            return 0
        except Exception as e:
            print(f"Error calculating production loss: {str(e)}")
            return 0
        """
        
        # For now, return 0 (no production loss data available)
        return 0

    def calculate_proper_severity_score(self, anomaly_scores):
        """
        Calculate a proper severity score from anomaly scores
        Returns a value between 0-100 representing severity percentage
        """
        if len(anomaly_scores) == 0:
            return 0
        
        # Get the average anomaly score for the day
        avg_score = np.mean(anomaly_scores)
        
        # Convert anomaly score to severity percentage
        # Isolation Forest scores are typically between -2 and 2
        # More negative = more anomalous
        
        if avg_score <= -1.5:  # Very severe
            severity = 85 + (abs(avg_score + 1.5) * 30)
        elif avg_score <= -1.0:  # Severe  
            severity = 60 + (abs(avg_score + 1.0) * 50)
        elif avg_score <= -0.5:  # Moderate
            severity = 35 + (abs(avg_score + 0.5) * 50)
        elif avg_score <= 0:     # Minor
            severity = 10 + (abs(avg_score) * 50)
        else:                    # Normal (shouldn't happen for anomalies)
            severity = max(0, 10 - (avg_score * 10))
        
        # Ensure we return an integer percentage between 0-100
        return min(max(int(severity), 0), 100)
        
    # Fix for the create_visualizations method to return a dictionary
    def create_visualizations(self, X_test, predictions, anomaly_scores, original_df):
        """Create visualizations and return as base64 encoded images"""
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Ellume Vision Analysis', fontsize=16, fontweight='bold')
        
        # 1. Anomaly score distribution
        axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=np.percentile(anomaly_scores, 5), color='red', linestyle='--',
                        label=f'5th percentile: {np.percentile(anomaly_scores, 5):.3f}')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Time series of total output power with anomalies highlighted
        if 'timestamp' in original_df.columns and 'total_output_power' in original_df.columns:
            axes[0, 1].plot(original_df['timestamp'], original_df['total_output_power'],
                        alpha=0.7, label='Normal', color='blue', linewidth=1)

            anomaly_mask = predictions == 1
            if np.any(anomaly_mask):
                axes[0, 1].scatter(original_df['timestamp'][anomaly_mask],
                                original_df['total_output_power'][anomaly_mask],
                                color='red', label='Anomalies', s=20, alpha=0.8)

            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Total Output Power')
            axes[0, 1].set_title('Power Output Over Time')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance (based on anomaly detection)
        if len(X_test.columns) > 0:
            # Calculate feature importance based on variance in anomaly scores
            feature_importance = []
            for col in X_test.columns:
                corr_with_score = np.corrcoef(X_test[col], anomaly_scores)[0, 1]
                feature_importance.append(abs(corr_with_score))

            top_features = sorted(zip(X_test.columns, feature_importance),
                                key=lambda x: x[1], reverse=True)[:10]

            features, importance = zip(*top_features)
            colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
            axes[1, 0].barh(range(len(features)), importance, color=colors)
            axes[1, 0].set_yticks(range(len(features)))
            axes[1, 0].set_yticklabels(features)
            axes[1, 0].set_xlabel('Correlation with Anomaly Score')
            axes[1, 0].set_title('Top 10 Most Important Features')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Anomaly detection results summary
        normal_count = np.sum(predictions == 0)
        anomaly_count = np.sum(predictions == 1)

        colors = ['#66c2a5', '#fc8d62']
        wedges, texts, autotexts = axes[1, 1].pie([normal_count, anomaly_count],
                    labels=['Normal', 'Anomaly'],
                    autopct='%1.1f%%',
                    colors=colors,
                    startangle=90)
        
        # Make the autotexts more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        axes[1, 1].set_title('Anomaly Detection Results')

        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Encode to base64
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Return as a dictionary with a single plot
        return {
            'main_analysis': plot_data
        }
    
    def create_lime_plot(self, lime_features, index):
        """Create a visualization for LIME feature contributions"""
        # Extract feature names and weights
        features = [item[0] for item in lime_features]
        weights = [item[1] for item in lime_features]
        
        # Create color mapping based on positive/negative weights
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Contribution Weight')
        ax.set_title(f'LIME Feature Contributions for Anomaly #{index+1}')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels to bars
        for i, (bar, weight) in enumerate(zip(bars, weights)):
            ax.text(weight + (0.01 if weight >= 0 else -0.01), i, 
                   f'{weight:.4f}', 
                   va='center', ha='left' if weight >= 0 else 'right',
                   fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Encode to base64
        plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return plot_data

# Routes
# 
# Add this route to your Flask app
@app.route('/<path:result_path>')
def show_result_by_path(result_path):
    """Display results by path (format: date_yymmdd_plant_id)"""
    try:
        # Query MongoDB for the document with this path
        if client is not None and collection is not None:
            document = collection.find_one({'path': result_path})
            
            if document:
                # Convert ObjectId to string for JSON serialization
                document['_id'] = str(document['_id'])
                
                # Prepare results for template
                results = {
                    'total_samples': document.get('total_samples', 0),
                    'anomaly_count': document.get('anomaly_count', 0),
                    'anomaly_percentage': document.get('anomaly_percentage', 0),
                    'plot_data': document.get('plot_data', {}),
                    'daily_summaries': document.get('summary', {}).get('daily_summaries', {}),
                    'severity_gauges': document.get('summary', {}).get('severity_gauges', {}),
                    'device_id': document.get('device_id', 'unknown'),
                    'date_yymmdd': document.get('date_yymmdd', 'unknown'),
                    'plant_id': document.get('plant_id', 'unknown'),
                    'path': document.get('path', ''),
                    'gauge_errors': document.get('summary', {}).get('gauge_errors', []),
                    'contamination': document.get('metadata', {}).get('contamination', 0.05),
                    'n_explanations': document.get('metadata', {}).get('n_explanations', 10)
                }
                
                # Add some dummy data for preview if not available
                if 'df_preview' not in results:
                    results['df_preview'] = [{'sample': 'data', 'value': 'preview'}]
                
                if 'anomaly_examples' not in results:
                    results['anomaly_examples'] = [{'sample': 'anomaly', 'value': 'example'}]
                
                return render_template('results.html', results=results)
            else:
                flash(f'No results found for path: {result_path}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Database connection not available', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        flash(f'Error retrieving results: {str(e)}', 'error')
        return redirect(url_for('index'))
    
@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    
    if form.validate_on_submit():
        try:
            # Save uploaded files
            csv_filename = secure_filename(form.csv_file.data.filename)
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
            form.csv_file.data.save(csv_path)
            
            # Load data
            df = pd.read_csv(csv_path)
            
            # Initialize detector with hardcoded model
            # Initialize detector with hardcoded model
            detector = SolarAnomalyDetector(
                contamination=form.contamination.data,
                groq_api_key=os.getenv('GEMINI_API_KEY')
            )
            
            # Preprocess data
            df_cleaned = detector.load_and_preprocess_data(df)
            df_processed = detector.feature_selection_and_engineering(df_cleaned)
            
            # Ensure the test features match training features
            X_test = df_processed[detector.feature_names]
            
            # Run predictions
            predictions, anomaly_scores, X_test_scaled = detector.predict_anomalies(X_test)
            
            # Add results to DataFrame
            df['anomaly'] = predictions
            df['anomaly_score'] = anomaly_scores
            
            # Get anomaly indices
            anomaly_indices = np.where(predictions == 1)[0]
            
            # Create visualizations (now returns dictionary)
            plot_data = detector.create_visualizations(X_test, predictions, anomaly_scores, df_cleaned)
            
            # Initialize LIME explainer if available
            if LIME_AVAILABLE:
                X_train_scaled = detector.scaler.transform(X_test)  # Using test data as proxy
                detector.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_scaled,
                    feature_names=detector.feature_names,
                    class_names=['Normal', 'Anomaly'],
                    mode='classification',
                    discretize_continuous=True
                )
            
            # Explain anomalies and generate daily summaries
            daily_summaries = detector.explain_anomalies_with_lime_and_groq(
                X_test=X_test,
                X_test_scaled=X_test_scaled,
                anomaly_indices=anomaly_indices,
                original_df=df_cleaned,
                n_explanations=10
            )
            
            # Debug: Check what's in daily_summaries
            print("Daily summaries type:", type(daily_summaries))
            print("Daily summaries content:", daily_summaries)
            
            # Ensure daily_summaries is a dictionary before processing
            if not isinstance(daily_summaries, dict):
                print(f"Warning: daily_summaries is not a dict, it's {type(daily_summaries)}")
                daily_summaries = {}
            
            # Create severity gauges for each device and date
            severity_gauges = {}
            gauge_errors = []
            
            for device_id, dates in daily_summaries.items():
                print(f"Processing severity gauge for device: {device_id}")
                
                if not isinstance(dates, dict):
                    print(f"Warning: dates for {device_id} is not a dict, it's {type(dates)}")
                    continue
                    
                severity_gauges[device_id] = {}
                
                for date_str, summary_data in dates.items():
                    print(f"Processing severity gauge for {device_id} on {date_str}")
                    
                    if not isinstance(summary_data, dict):
                        print(f"Warning: summary_data for {device_id} {date_str} is not a dict, it's {type(summary_data)}")
                        continue
                    
                    # Check if the required key exists
                    if 'avg_severity' not in summary_data:
                        print(f"Warning: 'avg_severity' not found in summary_data for {device_id} {date_str}")
                        print(f"Available keys: {list(summary_data.keys())}")
                        continue
                    
                    try:
                        severity_score = summary_data['avg_severity']
                        print(f"Creating gauge for {device_id} {date_str} with severity: {severity_score}")
                        
                        gauge_image = detector.create_severity_gauge(severity_score, device_id, date_str)
                        
                        if gauge_image is not None:
                            severity_gauges[device_id][date_str] = gauge_image
                            print(f"Successfully created gauge for {device_id} {date_str}")
                        else:
                            print(f"Failed to create gauge for {device_id} {date_str}")
                            gauge_errors.append(f"{device_id} {date_str}: Gauge creation returned None")
                            
                    except Exception as gauge_error:
                        error_msg = f"Error creating gauge for {device_id} {date_str}: {str(gauge_error)}"
                        print(error_msg)
                        gauge_errors.append(error_msg)
            
            # Print summary of gauge creation
            print(f"Successfully created {sum(len(dates) for dates in severity_gauges.values())} severity gauges")
            if gauge_errors:
                print(f"Gauge creation errors: {gauge_errors}")
            
            # In your index() route, after processing:
            if daily_summaries:
                # Get the first device and its first date from daily_summaries
                device_id = next(iter(daily_summaries.keys()), 'unknown')
                if device_id != 'unknown':
                    # Get the first date for this device
                    date_str = next(iter(daily_summaries[device_id].keys()), 'unknown')
                    
                    # Convert the date string to yymmdd format
                    try:
                        # Parse the date string (assuming format like "2024-01-15")
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        date_yymmdd = date_obj.strftime('%y%m%d')
                    except:
                        # If parsing fails, use current date as fallback
                        date_yymmdd = datetime.now().strftime('%y%m%d')
                else:
                    date_yymmdd = datetime.now().strftime('%y%m%d')
            else:
                device_id = 'unknown'
                date_yymmdd = datetime.now().strftime('%y%m%d')

            # Generate path in the format: date_yymmdd_device_id
            path = f"{date_yymmdd}_lPZ7JuDDGjzWmPATqJEu"

            # Prepare results for template
            results = {
                'total_samples': int(len(df)),
                'anomaly_count': int(len(anomaly_indices)),
                'anomaly_percentage': float((len(anomaly_indices) / len(df)) * 100),
                'plot_data': plot_data,
                'daily_summaries': daily_summaries,
                'severity_gauges': severity_gauges,
                'df_preview': convert_numpy_types(df.head(10).to_dict('records')),
                'anomaly_examples': convert_numpy_types(df[df['anomaly'] == 1].head(5).to_dict('records')),
                'gauge_errors': gauge_errors if gauge_errors else None,
                'device_id': device_id,
                'date_yymmdd': date_yymmdd,
                'plant_id': 'lPZ7JuDDGjzWmPATqJEu',
                'path': path,
                'contamination': form.contamination.data,
                'n_explanations': form.n_explanations.data
            }
            
            # Convert any remaining NumPy types in the results
            results = convert_numpy_types(results)
            
            # PUSH RESULTS TO MONGODB
            document_id = push_to_mongodb(results, csv_filename)
            if document_id:
                results['mongodb_id'] = document_id
                flash('Results successfully saved to database!', 'success')
            else:
                flash('Results processed but could not save to database', 'warning')
            # In your index() route, after processing and saving to MongoDB:
            
            # For session storage, use a minimal approach
            session['results_available'] = True
            session['results_summary'] = {
                'total_samples': int(len(df)),
                'anomaly_count': int(len(anomaly_indices)),
                'anomaly_percentage': float((len(anomaly_indices) / len(df)) * 100),
                'mongodb_id': document_id
            }
            
            # Save the full DataFrame to a temporary file instead of session
            temp_dir = tempfile.mkdtemp()
            results_file = os.path.join(temp_dir, 'anomaly_results.csv')
            df.to_csv(results_file, index=False)
            
            # Store the file path in session (not the actual data)
            session['results_file'] = results_file
            # In your index() route, after processing and saving to MongoDB:
            if document_id:
                results['mongodb_id'] = document_id
                flash('Results successfully saved to database!', 'success')
                
                # Redirect to the path URL instead of rendering results.html directly
                return redirect(url_for('show_result_by_path', result_path=path))
            else:
                flash('Results processed but could not save to database', 'warning')
                return render_template('results.html', results=results, form=form)
            return render_template('results.html', results=results, form=form)
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # This will print the full traceback
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return render_template('index.html', form=form)

@app.route('/download_results')
def download_results():
    """Download results as CSV"""
    try:
        results_file = session.get('results_file')
        if not results_file or not os.path.exists(results_file):
            flash('No results to download', 'error')
            return redirect(url_for('index'))
        
        # Clean up the file after sending
        response = send_file(
            results_file,
            mimetype='text/csv',
            as_attachment=True,
            download_name='solar_anomaly_results.csv'
        )
        
        # Schedule file cleanup after response
        @response.call_on_close
        def cleanup_file():
            try:
                if os.path.exists(results_file):
                    os.remove(results_file)
                    temp_dir = os.path.dirname(results_file)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
            except:
                pass
        
        return response
        
    except Exception as e:
        flash(f'Error downloading results: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for programmatic analysis"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Initialize detector
        # Initialize detector
        detector = SolarAnomalyDetector(
            contamination=data.get('contamination', 0.05),
            groq_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Preprocess and predict
        df_cleaned = detector.load_and_preprocess_data(df)
        df_processed = detector.feature_selection_and_engineering(df_cleaned)
        X_test = df_processed[detector.feature_names]
        predictions, anomaly_scores, _ = detector.predict_anomalies(X_test)
        
        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_count': int(np.sum(predictions == 1)),
            'total_samples': len(predictions)
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/results')
def list_results():
    """List all available results in the database"""
    try:
        if client is None or collection is None:
            flash('Database connection not available', 'error')
            return redirect(url_for('index'))
        
        # Get all documents from MongoDB
        documents = list(collection.find().sort('timestamp', -1))
        
        # Prepare data for template
        results_list = []
        for doc in documents:
            results_list.append({
                'path': doc.get('path', ''),
                'filename': doc.get('filename', 'unknown'),
                'timestamp': doc.get('timestamp', ''),
                'device_id': doc.get('device_id', 'unknown'),
                'date': doc.get('date_yymmdd', 'unknown'),
                'anomaly_count': doc.get('anomaly_count', 0),
                'total_samples': doc.get('total_samples', 0)
            })
        
        return render_template('results_list.html', results_list=results_list)
        
    except Exception as e:
        flash(f'Error retrieving results list: {str(e)}', 'error')
        return redirect(url_for('index'))
    
@app.route('/api/results/<document_id>')
def get_results(document_id):
    """API endpoint to retrieve results from MongoDB"""
    try:
        document = get_from_mongodb(document_id)
        if document:
            return jsonify(document)
        else:
            return jsonify({'error': 'Document not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)