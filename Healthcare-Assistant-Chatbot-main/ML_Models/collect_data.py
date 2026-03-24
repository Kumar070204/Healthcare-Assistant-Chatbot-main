import requests
import zipfile
import os
import json
from bs4 import BeautifulSoup
from datasets import load_dataset
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")

def collect_medquad():
    logger.info("Collecting MedQuAD...")
    try:
        ensure_dir("data/raw/medquad")
        url = "https://github.com/abachaa/MedQuAD/archive/refs/heads/master.zip"
        response = requests.get(url)
        with open("data/raw/medquad.zip", "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile("data/raw/medquad.zip", "r") as zip_ref:
            zip_ref.extractall("data/raw/medquad")
        os.remove("data/raw/medquad.zip")
        logger.info("MedQuAD collected successfully")
    except Exception as e:
        logger.error(f"Error collecting MedQuAD: {e}")

def collect_healthcaremagic():
    logger.info("Collecting HealthcareMagic...")
    try:
        ensure_dir("data/raw")
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        df = pd.DataFrame(dataset['train'])
        df[['Question']].to_csv("data/raw/healthcaremagic_questions.csv", index=False)
        logger.info("HealthcareMagic collected successfully")
    except Exception as e:
        logger.error(f"Error collecting HealthcareMagic: {e}")

def collect_who_myths():
    logger.info("Collecting WHO Mythbusters...")
    try:
        ensure_dir("data/raw")
        url = "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        myths = []
        # Adjust selector based on WHO site (inspect webpage)
        for myth in soup.find_all('div', class_='sf-content-block'):
            claim = myth.find('h3').text.strip() if myth.find('h3') else "Unknown"
            info = myth.find('p').text.strip() if myth.find('p') else "No info"
            myths.append({"claim": claim, "status": "False", "info": info, "source": "WHO"})
        with open("data/raw/who_myths.json", "w") as f:
            json.dump(myths, f)
        logger.info("WHO Mythbusters collected successfully")
    except Exception as e:
        logger.error(f"Error collecting WHO Mythbusters: {e}")

def collect_cdc_myths():
    logger.info("Collecting CDC Myths...")
    try:
        ensure_dir("data/raw")
        url = "https://www.cdc.gov/flu/about/myths.htm"  # Adjust URL
        # Implement similar scraping logic; placeholder
        myths = []  # Add actual scraping
        with open("data/raw/cdc_myths.json", "w") as f:
            json.dump(myths, f)
        logger.info("CDC Myths collected successfully")
    except Exception as e:
        logger.error(f"Error collecting CDC Myths: {e}")

def collect_unicef_myths():
    logger.info("Collecting UNICEF Myths...")
    try:
        ensure_dir("data/raw")
        url = "https://www.unicef.org/immunization/myths-and-facts"
        # Implement similar scraping logic
        myths = []  # Add actual scraping
        with open("data/raw/unicef_myths.json", "w") as f:
            json.dump(myths, f)
        logger.info("UNICEF Myths collected successfully")
    except Exception as e:
        logger.error(f"Error collecting UNICEF Myths: {e}")

def collect_health_feedback():
    logger.info("Collecting Health Feedback...")
    try:
        ensure_dir("data/raw")
        url = "https://healthfeedback.org/claim-reviews/"
        # Implement similar scraping logic
        myths = []  # Add actual scraping
        with open("data/raw/health_feedback.json", "w") as f:
            json.dump(myths, f)
        logger.info("Health Feedback collected successfully")
    except Exception as e:
        logger.error(f"Error collecting Health Feedback: {e}")

def build_mb_kb():
    logger.info("Building MB-KB...")
    try:
        ensure_dir("data/processed")
        mb_kb = []
        for file in ["data/raw/who_myths.json", "data/raw/cdc_myths.json", 
                    "data/raw/unicef_myths.json", "data/raw/health_feedback.json"]:
            if os.path.exists(file):
                with open(file, "r") as f:
                    mb_kb.extend(json.load(f))
        with open("data/processed/mb_kb.json", "w") as f:
            json.dump(mb_kb, f)
        logger.info("MB-KB built successfully")
    except Exception as e:
        logger.error(f"Error building MB-KB: {e}")

if __name__ == "__main__":
    logger.info("Starting data collection...")
    collect_medquad()
    collect_healthcaremagic()
    collect_who_myths()
    collect_cdc_myths()
    collect_unicef_myths()
    collect_health_feedback()
    build_mb_kb()
    logger.info("Data collection completed")