import logging
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from rag_query import RAGPipeline
import io
import base64
import os
import pandas as pd
import importlib.util
import re
import pickle
from PyPDF2 import PdfReader
import time
import numpy as np
from rapidfuzz import fuzz, process
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app with explicit template and static folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'), static_folder=os.path.join(BASE_DIR, 'static'))

# Debug print to verify paths
print(f"Base directory: {BASE_DIR}")
print(f"Template folder: {os.path.join(BASE_DIR, 'templates')}")
print(f"Static folder: {os.path.join(BASE_DIR, 'static')}")

# Initialize RAGPipeline
try:
    rag = RAGPipeline()
except Exception as e:
    logger.error(f"Failed to initialize RAGPipeline: {e}")
    raise

# =========================
# Stroke Prediction Setup
# =========================
STROKE_MODEL_PATH = os.path.join(BASE_DIR, 'best_model_resnet_neurology.pth')
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STROKE_CLASS_NAMES = ['Normal', 'Stroke']

# Data transforms for stroke image preprocessing
stroke_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load stroke prediction model
try:
    stroke_model = models.resnet50(pretrained=False)
    stroke_model.fc = nn.Linear(stroke_model.fc.in_features, 2)  # 2 classes: Normal / Stroke
    stroke_model.load_state_dict(torch.load(STROKE_MODEL_PATH, map_location=DEVICE))
    stroke_model = stroke_model.to(DEVICE)
    stroke_model.eval()
    logger.info("Stroke prediction model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load stroke prediction model: {e}")
    stroke_model = None

# =========================
# Eye Disease Prediction Setup
# =========================
EYE_MODEL_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\best_eye_disease_model.pth"
EYE_CLASS_NAMES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Data transforms for eye image preprocessing
eye_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load eye disease prediction model
try:
    eye_model = models.resnet50(weights=None)
    eye_model.fc = nn.Linear(eye_model.fc.in_features, len(EYE_CLASS_NAMES))
    eye_model.load_state_dict(torch.load(EYE_MODEL_PATH, map_location=DEVICE))
    eye_model = eye_model.to(DEVICE)
    eye_model.eval()
    logger.info("Eye disease prediction model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load eye disease model: {e}")
    eye_model = None

def load_module_from_path(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# =========================
# Food Recommendation Setup
# =========================
FOOD_REC_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\food_rec.py"
FOOD_DATA_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\Indian_Food_Nutrition_Categorized_Broad_with_labels.csv"

try:
    food_rec_module = load_module_from_path(FOOD_REC_PATH)
    logger.info("Food recommendation module loaded dynamically")
except Exception as e:
    logger.error(f"Failed to load food recommendation module: {e}")
    food_rec_module = None

try:
    food_df = pd.read_csv(FOOD_DATA_PATH)
    food_df.fillna('', inplace=True)
    # Normalize and prepare fields
    if 'machine_keywords' not in food_df.columns:
        food_df['machine_keywords'] = food_df['Diseases_Symptoms'].astype(str).str.lower().str.replace(r'[^a-z0-9|_ ]','',regex=True)
    food_df['machine_keywords'] = food_df['machine_keywords'].astype(str).str.lower()
    food_df['dish_name'] = food_df.get('Dish Name', food_df.columns[0]).astype(str)
    descr_cols = [c for c in ['Broad_Category','Detailed_Category','Detailed Category','Detailed_Category.1'] if c in food_df.columns]
    food_df['short_desc'] = food_df[descr_cols].astype(str).agg(' | '.join, axis=1)
    logger.info("Food recommendation dataset loaded successfully")
except Exception as e:
    logger.error(f"Failed to load food recommendation dataset: {e}")
    food_df = None

def predict_image(image_data, model_type="stroke"):
    """
    Predict disease from an image based on the specified model type.
    
    Args:
        image_data: Base64 encoded image string
        model_type: "stroke" or "eye" to select the appropriate model
    
    Returns:
        dict: Prediction result
    """
    if model_type == "stroke" and stroke_model is None:
        return {"error": "Sorry, the stroke prediction model couldn't be loaded. Please check the model file."}
    elif model_type == "eye" and eye_model is None:
        return {"error": "Sorry, the eye disease prediction model couldn't be loaded. Please check the model file."}

    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Select transform and model based on type
        transform = stroke_transform if model_type == "stroke" else eye_transform
        model = stroke_model if model_type == "stroke" else eye_model
        class_names = STROKE_CLASS_NAMES if model_type == "stroke" else EYE_CLASS_NAMES

        # Preprocess image
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[pred_idx]
            confidence = probs[0][pred_idx].item()

        # Convert probabilities to native Python floats
        probabilities = {cls: float(prob * 100) for cls, prob in zip(class_names, probs[0].cpu().numpy())}

        return {"prediction": f"The scan looks {pred_class.lower()} with {confidence*100:.1f}% confidence.",
                "probabilities": probabilities
               }
    except Exception as e:
        logger.error(f"Error processing {model_type} image: {e}")
        return {"error": f"Oops, something went wrong: {str(e)}"}

def process_query(query, mode):
    """
    Process the user query using the RAG pipeline with the selected mode.
    
    Args:
        query (str): The user's query
        mode (str): The mode (qa, fact_check, myth)
    
    Returns:
        dict: Response from RAG pipeline
    """
    logger.info(f"Processing query: {query} (mode: {mode})")
    if not query.strip():
        return {"response": "Oops, please type a question!"}
    try:
        response = rag.process_query(query, mode.lower())
        return {"response": response}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"response": f"Sorry, something went wrong: {str(e)}"}

# Load hospitals dataset globally
HOSPITALS_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\Patient_Appointment\data\hospitals.csv"
try:
    hospitals_df = pd.read_csv(HOSPITALS_PATH)
    logger.info("Hospitals dataset loaded successfully")
    required_columns = ['Doctor_ID', 'Doctor_Name', 'Specialization', 'Experience_Years', 'Contact_No',
                       'Hospital_ID', 'Hospital_Name', 'Area', 'City', 'Rating', 'Lat', 'Lng']
    if not all(col in hospitals_df.columns for col in required_columns):
        logger.error("Dataset missing required columns")
        hospitals_df = None
except FileNotFoundError:
    logger.error("hospitals.csv not found at the specified path")
    hospitals_df = None
except Exception as e:
    logger.error(f"Error loading hospitals dataset: {e}")
    hospitals_df = None

# Dynamically load functions from provided file paths
GEOLOCATION_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\Patient_Appointment\utils\geolocation.py"
DISTANCE_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\Patient_Appointment\utils\distance.py"
DISEASE_PREDICTION_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\disease_prediction.py"

def load_module_from_path(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

try:
    geolocation_module = load_module_from_path(GEOLOCATION_PATH)
    get_user_location = geolocation_module.get_user_location
    logger.info("Geolocation module loaded dynamically")
except Exception as e:
    logger.error(f"Failed to load geolocation module: {e}")
    get_user_location = None

try:
    distance_module = load_module_from_path(DISTANCE_PATH)
    haversine = distance_module.haversine
    logger.info("Distance module loaded dynamically")
except Exception as e:
    logger.error(f"Failed to load distance module: {e}")
    haversine = None

try:
    disease_prediction_module = load_module_from_path(DISEASE_PREDICTION_PATH)
    logger.info("Disease prediction module loaded dynamically")
except Exception as e:
    logger.error(f"Failed to load disease prediction module: {e}")

def find_nearest_hospital(specialization, user_lat, user_lng):
    if hospitals_df is None:
        return {"error": "Hospitals dataset couldn't be loaded."}
    
    if haversine is None:
        return {"error": "Haversine function not available."}
    
    filtered_df = hospitals_df[hospitals_df['Specialization'].str.lower() == specialization.lower()].copy()

    if filtered_df.empty:
        return {"error": f"No doctors found with specialization: {specialization}"}

    filtered_df.loc[:, 'Distance_km'] = filtered_df.apply(
        lambda row: haversine(user_lat, user_lng, row['Lat'], row['Lng']),
        axis=1
    )
    logger.info("Distances calculated successfully")

    nearest = filtered_df.sort_values(by='Distance_km').iloc[0]

    result = {
        "hospital_name": nearest['Hospital_Name'],
        "area": nearest['Area'],
        "city": nearest['City'],
        "doctor_name": nearest['Doctor_Name'],
        "doctor_id": str(nearest['Doctor_ID']),
        "specialization": nearest['Specialization'],
        "experience": int(nearest['Experience_Years']),
        "rating": float(nearest['Rating']),
        "contact": nearest['Contact_No'],
        "distance": nearest['Distance_km']
    }

    return result

# =========================
# Eligibility Prediction Setup
# =========================
ELIGIBILITY_MODEL_PATH = r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\db\eligibility_model.pkl"

try:
    with open(ELIGIBILITY_MODEL_PATH, "rb") as f:
        eligibility_model = pickle.load(f)
    logger.info("Eligibility model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load eligibility model: {e}")
    eligibility_model = None

def extract_text_from_pdf(pdf_data):
    """Read all text from PDF"""
    text = ""
    reader = PdfReader(io.BytesIO(pdf_data))
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def extract_condition_from_bill(text):
    """Find 'Condition/Feature' in hospital bill"""
    m = re.search(r"condition/feature\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).split("\n")[0].strip()
    return None

def predict_eligibility(pdf_data):
    if eligibility_model is None:
        return {"error": "Eligibility model couldn't be loaded."}
    
    try:
        bill_text = extract_text_from_pdf(pdf_data)
        condition = extract_condition_from_bill(bill_text)
        
        if not condition:
            return {"error": "Could not extract Condition/Feature from PDF"}
        
        prediction = eligibility_model.predict([condition])[0]
        
        result = {
            "extracted_condition": condition,
            "predicted_eligibility": prediction
        }
        return result
    except Exception as e:
        logger.error(f"Error in eligibility prediction: {e}")
        return {"error": f"Oops, something went wrong: {str(e)}"}

# =========================
# Disease Prediction Setup
# =========================
def get_disease_prediction(session_id, main_problem, answer):
    try:
        module = load_module_from_path(DISEASE_PREDICTION_PATH)
        dataset = pd.read_csv(r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\akiantor_generated_dataset.csv")
        condition_col = dataset.columns[0]
        dataset = dataset.rename(columns={condition_col: "condition"})
        questions = pd.read_csv(r"C:\Users\Kumaraswamy\Downloads\Data_Quest (2)\Data_Quest\scripts\data\akiantor_questions.csv")

        if session_id not in getattr(app, 'disease_sessions', {}):
            app.disease_sessions = getattr(app, 'disease_sessions', {})
            app.disease_sessions[session_id] = module.AnchoredAkinator(dataset, questions, main_problem)
        aki = app.disease_sessions[session_id]

        # If an answer was supplied, apply it to the last asked symptom so
        # we don't accidentally apply the answer to a different symptom.
        if answer is not None:
            a = str(answer).strip().lower()
            logger.info(f"Updating with answer for last asked: {a}")
            aki.update(aki.last_asked, a)

        next_symptom = aki.get_next_symptom()
        logger.info(f"Next symptom: {next_symptom}, Num questions: {aki.num_questions}, Available symptoms: {len(aki.available_symptoms) if hasattr(aki, 'available_symptoms') else 'N/A'}")
        # Only produce a question if ask_question yields text; otherwise go to prediction
        if next_symptom is not None:
            question_text = aki.ask_question(next_symptom)
            if question_text:
                return {"question": f"Q{aki.num_questions + 1}: {question_text}", "session_id": session_id}

        # No further meaningful questions — produce prediction
        prediction = aki.predict()
        precautions = aki.precautions(prediction) if prediction else []
        logger.info(f"Prediction made: {prediction}, Precautions: {precautions}")
        return {"prediction": prediction, "precautions": precautions, "done": True}
    except Exception as e:
        logger.error(f"Error in disease prediction: {e}")
        return {"error": f"Oops, something went wrong: {str(e)}"}

# =========================
# Food Recommendation Logic
# =========================
def get_food_recommendation(session_id, user_input, action=None):
    if food_rec_module is None or food_df is None:
        return {"error": "Food recommendation module or dataset couldn't be loaded."}
    
    try:
        if session_id not in getattr(app, 'food_sessions', {}):
            app.food_sessions = getattr(app, 'food_sessions', {})
            app.food_sessions[session_id] = food_rec_module.RecommenderState()
        
        state = app.food_sessions[session_id]
        
        if action == 'more':
            recs = food_rec_module.recommend_next(top_k=5)
            if not recs:
                return {"message": "No more recommendations available.", "done": False}
            message = "Next recommendations:\n" + "\n".join([f"{r['rank']}. {r['dish']} -- {r['explanation']}" for r in recs])
            return {"message": message, "done": False}
        
        if action == 'exit':
            if session_id in getattr(app, 'food_sessions', {}):
                del app.food_sessions[session_id]
            return {"message": "Exiting. Enjoy your meal!", "done": True}
        
        if not user_input:
            return {"message": "Please tell me what you'd like to eat to get recommendations.", "done": False}
        
        recs = food_rec_module.recommend(user_input, top_k=5)
        if not recs:
            return {"message": "Sorry, I couldn't find any recommendations based on your request.", "done": False}
        
        message = "Top recommendations for you:\n" + "\n".join([f"{r['rank']}. {r['dish']} -- {r['explanation']}" for r in recs])
        return {"message": message, "done": False}
    except Exception as e:
        logger.error(f"Error in food recommendation: {e}")
        return {"error": f"Oops, something went wrong: {str(e)}"}

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('query', '')
    mode = data.get('mode', 'qa')
    return jsonify(process_query(query, mode))

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image', '')
    return jsonify(predict_image(image_data, model_type="stroke"))

@app.route('/api/eye_predict', methods=['POST'])
def eye_predict():
    data = request.get_json()
    image_data = data.get('image', '')
    return jsonify(predict_image(image_data, model_type="eye"))

@app.route('/api/appointment', methods=['POST'])
def appointment():
    data = request.get_json()
    specialization = data.get('specialization', 'Pediatrics')
    lat = data.get('lat')
    lng = data.get('lng')
    if lat is None or lng is None:
        if get_user_location is None:
            return jsonify({"error": "Geolocation function not available and location not provided"})
        try:
            lat, lng = get_user_location()
            logger.info(f"Using geolocation: Lat={lat}, Lng={lng}")
        except Exception as e:
            logger.error(f"Failed to get user location: {e}")
            return jsonify({"error": f"Location not provided and geolocation failed: {str(e)}"})
    try:
        result = find_nearest_hospital(specialization, float(lat), float(lng))
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in appointment: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/eligibility', methods=['POST'])
def eligibility():
    data = request.get_json()
    pdf_data = data.get('pdf', '')
    if not pdf_data:
        return jsonify({"error": "No PDF data provided"})
    try:
        pdf_bytes = base64.b64decode(pdf_data.split(',')[1])
        result = predict_eligibility(pdf_bytes)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in eligibility endpoint: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/disease', methods=['POST'])
def disease():
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    main_problem = data.get('main_problem')
    answer = data.get('answer')
    if not main_problem and not answer:
        return jsonify({"message": "🧠 Medical Akinator Chatbot\nI’ll ask yes/no questions to narrow down your condition.\nReply with: yes / no / nothing else. Type 'exit' anytime.\nChatbot: Please describe your main problem.", "session_id": session_id})
    if main_problem:
        return jsonify({"message": f"Chatbot: Got it. Let's explore further...\nQ1: {get_disease_prediction(session_id, main_problem, None)['question'].split(': ')[1]}", "session_id": session_id})
    result = get_disease_prediction(session_id, None, answer)
    return jsonify(result)

@app.route('/api/food', methods=['POST'])
def food_recommendation():
    data = request.get_json()
    session_id = data.get('session_id', 'default')
    user_input = data.get('user_input')
    action = data.get('action')
    if not user_input and not action:
        return jsonify({"message": "Please tell me what you'd like to eat to get recommendations.", "session_id": session_id, "done": False})
    result = get_food_recommendation(session_id, user_input, action)
    result['session_id'] = session_id
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)