import xml.etree.ElementTree as ET
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_medquad():
    logger.info("Parsing MedQuAD XML...")
    try:
        qa_pairs = []
        # Walk through all XML files in data/raw/medquad
        for root, _, files in os.walk("data/raw/medquad"):
            for file in files:
                if file.endswith(".xml"):
                    file_path = os.path.join(root, file)
                    logger.info(f"Processing file: {file_path}")
                    try:
                        tree = ET.parse(file_path)
                        xml_root = tree.getroot()
                        for qa in xml_root.findall(".//QAPair"):
                            question_elem = qa.find("Question")
                            answer_elem = qa.find("Answer")
                            # Use explicit checks to avoid deprecation warnings
                            question = question_elem.text.strip() if question_elem is not None and question_elem.text else ""
                            answer = answer_elem.text.strip() if answer_elem is not None and answer_elem.text else ""
                            if question and answer:
                                qa_pairs.append({"question": question, "answer": answer})
                    except ET.ParseError as e:
                        logger.error(f"Error parsing {file_path}: {e}")
        if not qa_pairs:
            logger.warning("No Q&A pairs found in MedQuAD XML files")
        df = pd.DataFrame(qa_pairs)
        os.makedirs("data/raw", exist_ok=True)
        df.to_csv("data/raw/medquad.csv", index=False)
        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to data/raw/medquad.csv")
        return df
    except Exception as e:
        logger.error(f"Error parsing MedQuAD: {e}")
        return None

if __name__ == "__main__":
    parse_medquad()