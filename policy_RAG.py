import openai
import fitz  # PyMuPDF for PDF text extraction
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Set up your OpenAI API key
openai.api_key = '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'

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")
    return text

# Chunk text into smaller pieces based on token limit
def chunk_text(text, max_tokens=3000):
    sentences = text.split(". ")
    chunks = []
    chunk = []
    tokens_count = 0

    for sentence in sentences:
        tokens = len(sentence.split())
        if tokens_count + tokens <= max_tokens:
            chunk.append(sentence)
            tokens_count += tokens
        else:
            chunks.append(". ".join(chunk))
            chunk = [sentence]
            tokens_count = tokens

    if chunk:
        chunks.append(". ".join(chunk))
    return chunks

# Generate embeddings for a chunk of text
def get_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response['data'][0]['embedding'])

# Store each chunk with its embedding for retrieval
def create_embeddings_store(chunks):
    return [{"text": chunk, "embedding": get_embeddings(chunk)} for chunk in chunks]

# Retrieve relevant chunks based on similarity
def retrieve_relevant_chunks(query, embedding_store, top_k=3):
    query_embedding = get_embeddings(query)
    similarities = [
        cosine_similarity([query_embedding], [doc["embedding"]])[0][0]
        for doc in embedding_store
    ]
    relevant_indices = np.argsort(similarities)[-top_k:][::-1]
    return [embedding_store[i]["text"] for i in relevant_indices]

# Load, chunk, and embed each policy
company_policy_text = extract_text_from_pdf("national_policies/company_policy.pdf")
google_policy_text = extract_text_from_pdf("google_policies/google_policy.pdf")
microsoft_policy_text = extract_text_from_pdf("google_policies/microsoft_policy.pdf")

# Chunk and create embedding stores for each policy
company_policy_chunks = chunk_text(company_policy_text)
google_policy_chunks = chunk_text(google_policy_text)
microsoft_policy_chunks = chunk_text(microsoft_policy_text)

company_policy_store = create_embeddings_store(company_policy_chunks)
google_policy_store = create_embeddings_store(google_policy_chunks)
microsoft_policy_store = create_embeddings_store(microsoft_policy_chunks)

# Define the query (e.g., "Find contradictions in data usage policies")
query = "Identify contradictions in data sharing and privacy policies."

# Retrieve relevant chunks based on the query
relevant_company_chunks = retrieve_relevant_chunks(query, company_policy_store)
relevant_google_chunks = retrieve_relevant_chunks(query, google_policy_store)
relevant_microsoft_chunks = retrieve_relevant_chunks(query, microsoft_policy_store)

# Step 2: Define the Prompt for OpenAI API using retrieved chunks
prompt = f"""
You are a policy analyst. Compare the following sections of the policies and identify contradictions, if any, along with explanations.

### Relevant Company Policy Sections:
{''.join(relevant_company_chunks)}

### Relevant Google Policy Sections:
{''.join(relevant_google_chunks)}

### Relevant Microsoft Policy Sections:
{''.join(relevant_microsoft_chunks)}

Instructions:
- List contradictory points found between the company policy and the Google and Microsoft policies.
- Provide explanations for why these sections contradict.
"""

# Step 3: Send the Request to OpenAI API
# Step 3: Send the Request to OpenAI API with ChatCompletion.create
response = openai.ChatCompletion.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a policy analyst."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=1500,  # Adjust based on expected response length
    temperature=0.2
)

# Extract and print the response
comparison_result = response['choices'][0]['message']['content'].strip()
print(comparison_result)
