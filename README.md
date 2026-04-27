

# 🏥 Healthcare Assistant Chatbot  

A **multi-functional healthcare assistant chatbot** integrating **Q&A, fact-checking, myth-busting, symptom prediction, doctor appointment scheduling, personalized health tracking, medication management, insurance guidance, and diet recommendations**.  

---

## ✨ Features
1. **Q&A + Fact-Checker + Myth-Buster**  
   - Uses **MedQuAD dataset** for reliable answers  
   - Detects misinformation from **WHO, CDC, UNICEF**  
   - Busts myths with verified health facts  

2. **Personalized Health**  
   - Collects data from **wearables & manual inputs**  
   - Tracks vitals like **heart rate, steps, sleep, glucose**  
   - Provides **personalized recommendations**  

3. **Symptom Checker & Appointment**  
   - Symptom input & **ML/DL-based disease prediction**  
   - Suggests **doctors & hospitals** from local database  
   - Allows **appointment booking**  

4. **Medication Manager**  
   - Upload doctor prescriptions  
   - Stores medication history  
   - Sends **alerts/reminders** for medicines  

5. **Insurance Claim Assistance**  
   - User uploads insurance documents  
   - Bot guides on **claims, eligibility, process**  

6. **Food Recommendations**  
   - Suggests **diet plans** based on disease  
   - Supports **local/traditional diets**  

7. **Symptom Models**  
   - Separate **ML models** for:  
     - Dermatology 🧴  
     - Ophthalmology 👁️  
     - Cardiology ❤️  
     - Neurology 🧠  

8. **Extras**  
   - Multilingual support 🌍  
   - Works in **offline mode** with cached results  

---

## 📂 Folder Structure

├── app.py # Main chatbot app

├── collect_data.py # Wearable/manual data collection

├── disease_prediction.py # Symptom-based disease prediction

├── food_rec.py # Food recommendation system

├── interactive_query.py # Q&A + Fact-checking interface

├── parse_medquad.py # Parsing MedQuAD dataset

├── process_chunks.py # RAG preprocessing

├── rag_query.py # RAG-based query system

│

├── Patient_Appointment/

│ ├── app.py # Appointment scheduling app

│ ├── config.py # Configurations

│ ├── distance.py # Distance calculator for hospitals

│ ├── geolocation.py # Location-based hospital mapping

│ ├── hospitals.csv # Hospital-doctor database

│

├── DL_Disease_Prediction.ipynb # Deep Learning disease models

├── requirements.txt # Project dependencies

├── README.md # Project documentation


---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DataQuest_STJRK_06.git
   cd DataQuest_STJRK_06
Create a virtual environment and install dependencies:

bash
Copy code
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
🚀 Usage
Run the chatbot:

bash
Copy code
python app.py
Run appointment scheduling service:

bash
Copy code
cd Patient_Appointment
python app.py
Run disease prediction notebook:
Open DL_Disease_Prediction.ipynb in Jupyter Notebook or VSCode.

🧠 Tech Stack
Python, Node.js

RAG + Ollama (Medical-Llama3-8B) for Q&A

PyTorch / TensorFlow for ML models

Pandas, Scikit-learn for data processing

Flask / FastAPI for backend APIs

🔮 Future Enhancements
Add voice assistant for accessibility

Expand hospital & doctor DB with real data

Build mobile app integration

Enhance multilingual support

---

## Built by

**Kumaraswamy G** 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/kumaraswamy-g-872b81277/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kumaraswamy2004@gmail.com)

---
