import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
from transformers import pipeline
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Step 1: Load the Code of Ethics document
def load_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Step 2: Preprocess the text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize sentences
    sentences = sent_tokenize(text)
    return sentences

# Step 3: Topic Modeling Preparation
def prepare_topic_modeling(sentences):
    stop_words = set(stopwords.words('english'))
    processed_sentences = [[word for word in word_tokenize(sent) if word not in stop_words] for sent in sentences]
    return processed_sentences

# Step 4: Perform Topic Modeling
def perform_topic_modeling(processed_sentences, num_topics=3):
    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary(processed_sentences)
    corpus = [dictionary.doc2bow(sentence) for sentence in processed_sentences]
    # LDA Model
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model, corpus, dictionary

# Step 5: Perform Sentiment Analysis
def perform_sentiment_analysis(sentences):
    sentiment_pipeline = pipeline('sentiment-analysis')
    sentiments = sentiment_pipeline(sentences)
    return sentiments

# Step 6: Analyze and Output Results
def analyze_code_of_ethics(file_path):
    # Load and preprocess the text
    text = load_document(file_path)
    sentences = preprocess_text(text)
    
    # Topic Modeling
    print("Performing Topic Modeling...\n")
    processed_sentences = prepare_topic_modeling(sentences)
    lda_model, corpus, dictionary = perform_topic_modeling(processed_sentences, num_topics=3)
    
    # Display Topics
    print("Identified Topics:")
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"\nTopic {idx + 1}:")
        print(f"{topic}")
    print("\n")
    
    # Sentiment Analysis
    print("Performing Sentiment Analysis...\n")
    sentiments = perform_sentiment_analysis(sentences)
    sentiment_df = pd.DataFrame({
        'Sentence': sentences,
        'Sentiment': [sent['label'] for sent in sentiments],
        'Score': [sent['score'] for sent in sentiments]
    })
    
    # Display Sentiment Analysis Results
    print("Sentiment Analysis Results (Formatted with New Rows):\n")
    for index, row in sentiment_df.iterrows():
        print(f"Sentence: {row['Sentence']}\nSentiment: {row['Sentiment']}\nScore: {row['Score']:.4f}\n")
    
    # Save Results
    sentiment_df.to_csv("sentiment_analysis_results.csv", index=False)
    print("\nSentiment analysis results saved to 'sentiment_analysis_results.csv'.")

# Run the analysis
if __name__ == "__main__":
    file_path = "code_of_ethics.txt"  # Replace with your file path
    analyze_code_of_ethics(file_path)
