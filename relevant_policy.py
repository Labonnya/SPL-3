import openai
import os
import fitz  # PyMuPDF for PDF text extraction

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

# Load and chunk each policy
company_policy_path = "national_policies/company_policy.pdf"
google_policy_path = "google_policies/google_policy.pdf"
microsoft_policy_path = "google_policies/microsoft_policy.pdf"

company_policy_text = extract_text_from_pdf(company_policy_path)
google_policy_text = extract_text_from_pdf(google_policy_path)
microsoft_policy_text = extract_text_from_pdf(microsoft_policy_path)

# Chunk each policy to respect token limits
company_policy_chunks = chunk_text(company_policy_text)
google_policy_chunks = chunk_text(google_policy_text)
microsoft_policy_chunks = chunk_text(microsoft_policy_text)

# Open a file to store the results
output_file_path = "policy_comparison_results.txt"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    # Iterate over chunks and send to OpenAI API
    for i, company_chunk in enumerate(company_policy_chunks):
        for j, google_chunk in enumerate(google_policy_chunks):
            for k, microsoft_chunk in enumerate(microsoft_policy_chunks):
                # Construct a prompt for each combination of chunks
                prompt = f"""
                compare my company policy with microsoft and google policy and give those policies of my company that is contradictory with google and microsoft policies and also give reason why it is contradictory.

                ### Company Policy Section:
                {company_chunk}

                ### Google Policy Section:
                {google_chunk}

                ### Microsoft Policy Section:
                {microsoft_chunk}

                Instructions:
                - List contradictory points found between the company policy and the Google and Microsoft policies.
                - Provide explanations for why these sections contradict.
                """
                
                # Send the request to OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a policy analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500  # Adjust based on expected response length
                )

                # Collect the response
                result = response['choices'][0]['message']['content']

                # Append the result to the output file
                output_file.write(f"Comparison result for Company Section {i+1}, Google Section {j+1}, Microsoft Section {k+1}:\n")
                output_file.write(result + "\n\n")

                # Print progress to console
                print(f"Comparison result for Company Section {i+1}, Google Section {j+1}, Microsoft Section {k+1} written to file.")

print(f"\nAll results saved to {output_file_path}")
