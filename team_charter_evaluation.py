from transformers import pipeline
from nltk.tokenize import sent_tokenize
import re

# Step 1: Load the document
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Step 2: Define RAI-related keywords and phrases
rai_keywords = [
    "fairness", "accountability", "transparency", "bias mitigation",
    "privacy", "ethics", "diversity", "responsibility", "governance",
    "risk assessment", "stakeholder engagement", "explainability"
]

# Step 3: Analyze the document for RAI keywords
def analyze_rai_compliance(text, keywords):
    sentences = sent_tokenize(text)
    compliance_results = {}
    for keyword in keywords:
        compliance_results[keyword] = [sent for sent in sentences if keyword.lower() in sent.lower()]
    return compliance_results

# Step 4: Evaluate completeness and highlight gaps
def evaluate_compliance(compliance_results):
    total_keywords = len(rai_keywords)
    matched_keywords = sum(1 for key in compliance_results if compliance_results[key])
    score = (matched_keywords / total_keywords) * 100

    missing_keywords = [key for key in rai_keywords if not compliance_results[key]]
    return score, missing_keywords

# Main execution
if __name__ == "__main__":
    # Load the Team Charter document
    file_path = "team_charter.txt"  # Replace with your file path
    document_text = load_document(file_path)

    # Analyze the document
    compliance_results = analyze_rai_compliance(document_text, rai_keywords)

    # Evaluate the compliance
    score, missing_keywords = evaluate_compliance(compliance_results)

    # Output the results
    print(f"RAI Compliance Score: {score:.2f}%")
    if missing_keywords:
        print("Missing RAI elements in the document:")
        print(", ".join(missing_keywords))
    else:
        print("The Team Charter fully aligns with RAI principles.")
