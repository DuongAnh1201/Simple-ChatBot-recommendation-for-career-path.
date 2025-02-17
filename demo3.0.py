import pandas as pd
import openai
import faiss
from datasets import load_dataset
import numpy as np
import os

# Set your API keys here
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load dataset
dataset = load_dataset("lang-uk/recruitment-dataset-job-descriptions-english")

# Function to get text embeddings
def get_text_embedding(text):
    try:
        if not text.strip():  # Check if input is empty or just spaces
            print("Input text is empty.")
            return None

        # Ensure input is passed as a list of strings
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # Correct model name
            input=[text]  # Input should be a list of strings
        )

        # Extract the embedding data
        return response['data'][0]['embedding']
    
    except openai.error.InvalidRequestError as e:
        print(f"Invalid Request Error: {e}")
        return None

# Generate embeddings for each job description
recommendations = dataset['train'][:100]['Long Description']  # Access the correct field for job descriptions
embeddings = [get_text_embedding(recommendation) for recommendation in recommendations if get_text_embedding(recommendation) is not None]

# Convert embeddings to numpy array
embedding_matrix = np.array(embeddings).astype('float32')

# Build the FAISS index
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # Use L2 distance
index.add(embedding_matrix)

# Function to search similar reviews
def search_similar_reviews(query_text, top_k=3):
    # Generate embedding for the query
    query_embedding = np.array(get_text_embedding(query_text)).astype('float32').reshape(1, -1)
    if query_embedding is not None:
        distances, indices = index.search(query_embedding, top_k)
        return indices[0]  # Return only the indices
    return []

# Function to generate detailed responses using GPT
def generate_gpt_response(query_text, recommendations):
    # Custom prompt
    custom_prompt = f"""
    You are a highly knowledgeable and empathetic job recommendation supporter. Your goal is to assist users with their potential jobs and give them a hug with some lucky sentences.

    User: Hello, I am a {query_text}. I want to find a new job next year.
    Chatbot: Hi, it seems that you have experience as a {query_text}. Please wait for a moment while I find some job descriptions that might suit you.

    Here are some job descriptions and company names that might be suitable for you:
    """
    
    for idx in recommendations:
        job_description = dataset['train']['Long Description'][idx]
        company_name = dataset['train']['Company Name'][idx]
        position_name=dataset['train']['Position'][idx]
        custom_prompt += f"\n\nJob Description {idx + 1}:\nCompany Name: {company_name}\n{position_name}\n{job_description}\n"

    custom_prompt += "\nPlease provide detailed recommendations based on the above job descriptions. At the end, give them a hug by a lovely wish"

    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=custom_prompt,
        max_tokens=500,
        temperature=0.7,
        stop=["User:", "Chatbot:"]
    )

    return response.choices[0].text.strip()


# Example query
query_text = "data scientist"
results = search_similar_reviews(query_text)

# Print results with GPT-generated responses
if len(results)!=0:
    gpt_response = generate_gpt_response(query_text, results)
    print(f"GPT Response: {gpt_response}")
else:
    print("No similar job descriptions found.")
