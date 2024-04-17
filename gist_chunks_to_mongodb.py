
import os
import re
import csv
import certifi
import requests
import numpy as np
import pandas as pd
import pymongo
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Define the OpenAI API key here
openai.api_key = ''

def get_database():
    """
    Purpose: Establish a connection to the MongoDB database.
    """
    uri = "mongodb+srv://sriveerisetti:8TkNOyysCO4S3lBo@chatbot.w3bjnk6.mongodb.net/?retryWrites=true&w=majority&appName=Chatbot"
    ca = certifi.where()
    # The MongoClient is used to establish a connection to the database.
    client = MongoClient(uri, tlsCAFile=ca)
    # The database that contains the information is Chatbot
    db = client['Chatbot'] 
    return db

def embed_message(user_message):
    """
    Purpose: Embed the user's message using the GIST-large model.
    Input: user_message - The message that the user has entered.
    """
    # We ended up using Hugging Face's SentenceTransformer library to embed the user's message.
    # The GIST embedding model is on the leaderboard on Hugging Face
    model = SentenceTransformer("avsolatorio/GIST-large-Embedding-v0")
    # We use the encode functoin to embed the user's message
    query_embedding = model.encode([user_message], convert_to_tensor=True).tolist()[0]
    return query_embedding

def chunk_words(text, chunk_size=150, overlap=25):
    """
    Purpose: Split the text into chunks of a specified size with a specified overlap.
    Input: text - The text to split into chunks.
    Input: chunk_size - The size of each chunk (150).
    Input: overlap - The number of words to overlap between chunks (25).
    """
    # The purpose of this code is to make sure that the chunk size remains firmly at 150 words.
    # We use 25 words from the previous chunk to overlap with the next chunk and then 25 words from the next chunk to overlap with the previous chunk.
    # In total there are 150 words in each chunk.
    words = text.split()
    chunks = []
    start = 0 
    # Here we are making sure that the chunk size is 150 words.
    while start + chunk_size - 2 * overlap < len(words):
        # We are making sure that the start is at 0.
        if start == 0:
            actual_start = start
            # We are making sure that the end is at 150 words.
            actual_end = start + chunk_size - overlap  
        else:
            actual_start = start - overlap
            actual_end = start + chunk_size - overlap
        # Here we make sure tha the end is not greater than the length of the words (150)
        if actual_end + overlap > len(words):
            actual_end = len(words)  
        # Here we gather the words in the chunk.
        chunk = words[actual_start:actual_end]
        # Here we append the words in the chunk to the chunks list.
        chunks.append(' '.join(chunk))
        start += chunk_size - 2 * overlap
    # Here we are making sure that the last chunk is not greater than the length of the words.
    if start < len(words):
        # We use the max function to make sure of this.
        last_chunk_start = max(0, start - overlap)
        last_chunk = words[last_chunk_start:len(words)]
        chunks.append(' '.join(last_chunk))
    return chunks

def store_text_with_embedding(text, source, collection):
    """
    Store the text and its embedding in the database.
    :param text: The text to store
    :param source: The source of the text
    :param collection: The MongoDB collection in which to store the text
    """
    for chunk in chunk_words(text):
        chunk_embedding = embed_message(chunk)  # Use BERT-based embedding
        collection.insert_one({
            "chunk": chunk,
            "embedding": chunk_embedding,
            "source": source
        })
    print(f"Content from {source} has been successfully stored in MongoDB.")

def process_text_files(folder_path, collection):
    """
    Purpose: Process the text files in the specified folder and store the text and its embedding in the database.
    Input: folder_path - The path to the folder containing the text files.
    Input: collection - The MongoDB collection in which to store the text.
    """
    # We create a for loop that goes through the folder containing the text files.
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # For all files within the folder we read them and use the store_text_with_embedding function to store the text and its 
            # embedding in the database.
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
                store_text_with_embedding(text_content, filename, collection)

if __name__ == "__main__":
    db = get_database()
    collection = db['Duke5']
    folder_path = "/content/Combo_Data"
    process_text_files(folder_path, collection)
