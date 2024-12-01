

from sentence_transformers import SentenceTransformer, util
import os
import fitz  # PyMuPDF
import nltk

# Download NLTK data for sentence tokenization
nltk.download('punkt_tab')

# Load a pre-trained model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDFs
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Load policies from a folder, handling both text and PDF files
def load_policies_from_folder(folder_path):
    policies = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                policies[filename] = file.read()
        elif filename.endswith('.pdf'):
            policies[filename] = extract_text_from_pdf(file_path)
        else:
            print(f"Skipping unsupported file format: {filename}")
    return policies

# Split text into sentences
def split_into_sentences(text):
    return nltk.sent_tokenize(text)

# Load both sets of policies
company_policies = load_policies_from_folder('company_policies')
google_policies = load_policies_from_folder('google_policies')

# Define a threshold for low similarity
threshold = 0.3
contrasting_sentences = []

# Compare each sentence in company policies with each sentence in Google policies
for company_name, company_text in company_policies.items():
    company_sentences = split_into_sentences(company_text)
    company_embeddings = model.encode(company_sentences)
    
    for google_name, google_text in google_policies.items():
        google_sentences = split_into_sentences(google_text)
        google_embeddings = model.encode(google_sentences)
        
        # Compare each company sentence with each Google sentence
        for i, company_emb in enumerate(company_embeddings):
            for j, google_emb in enumerate(google_embeddings):
                similarity = util.cos_sim(company_emb, google_emb).item()
                if similarity < threshold:
                    contrasting_sentences.append((company_name, google_name, company_sentences[i], google_sentences[j], similarity))

# Output specific contrasting sentences
for company_name, google_name, company_sentence, google_sentence, similarity in contrasting_sentences:
    print(f"Contrasting sentences found:\nCompany Policy: {company_name}\nGoogle Policy: {google_name}\n")
    print(f"Company Statement: {company_sentence}")
    print(f"Google Statement: {google_sentence}")
    print(f"Similarity Score: {similarity:.4f}\n")
