from sentence_transformers import SentenceTransformer, util

# Step 1: Define Google and Microsoft Policy Statements
google_policy = """
Google AI Principles: Artificial intelligence should be socially beneficial, avoid creating or reinforcing bias, be built and tested for safety, and be accountable to people.
We strive to ensure AI systems are used responsibly and ethically.
"""

microsoft_policy = """
Microsoft Responsible AI: Our AI systems are developed to be fair, inclusive, transparent, secure, and accountable. 
We commit to empowering people with AI and ensuring compliance with privacy and security standards.
"""

# Step 2: Define the policy to compare (e.g., an organization's policy)
organization_policy = """
Our organization is committed to ethical AI practices, emphasizing fairness, transparency, and data privacy.
We address algorithmic bias and strive to develop AI systems that respect societal values and individual rights.
"""

# Step 3: Load Pre-trained Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 4: Encode Policies into Embeddings
google_embedding = model.encode(google_policy, convert_to_tensor=True)
microsoft_embedding = model.encode(microsoft_policy, convert_to_tensor=True)
organization_embedding = model.encode(organization_policy, convert_to_tensor=True)

# Step 5: Calculate Similarity Scores
google_similarity = util.pytorch_cos_sim(organization_embedding, google_embedding).item()
microsoft_similarity = util.pytorch_cos_sim(organization_embedding, microsoft_embedding).item()

# Step 6: Display Similarity Scores
print("Policy Similarity Scores:")
print(f"- Similarity with Google's AI Policy: {google_similarity:.2f}")
print(f"- Similarity with Microsoft's AI Policy: {microsoft_similarity:.2f}")
