
import os
import re
import csv
import certifi
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import openai

import configparser

config = configparser.ConfigParser()

config.read('configs/config.ini')

MONGO_DB = config['MONGO_DB']

SECRETS = config['SECRETS']

# Define the OpenAI API key here
openai.api_key = SECRETS['openai_api_key']

def get_database():
    """
    Purpose: Establish a connection to the MongoDB database.
    """
    uri = MONGO_DB['db_uri']
    ca = certifi.where()
    # The MongoClient is used to establish a connection to the database.
    client = MongoClient(uri, tlsCAFile=ca)
    # The database that contains the information is Chatbot
    db = client['Chatbot'] 
    return db

def generate_embedding(user_message):
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

def find_most_relevant_chunks(query, top_k=5):
    """
    Purpose: Find the most relevant chunks to the user's query.
    Input: query - the user's query
    Input: top_k - the number of most relevant chunks to return
    """
    # Here we connect to the database and also define the collection within the database 
    db = get_database()
    collection = db['Duke5']
    # We use the generate_embedding function to generate the embedding for the user's query
    query_embedding = np.array(generate_embedding(query)).reshape(1, -1)  
    docs = collection.find({})

    # Empty list to store the similarities
    similarities = []
    # For all chunks in the collection we calculate the cosine similarity between the query embedding and the document embedding
    for doc in docs:
        chunk_embedding = np.array(doc['embedding']).reshape(1, -1)  
        # We use the cosine similarity function to calculate the similarity between the query embedding and the document embedding
        similarity = cosine_similarity(chunk_embedding, query_embedding)[0][0]
        # We append the chunk, similarity and source to the similarities list
        similarities.append((doc['chunk'], similarity, doc.get('source')))

    # We sort the similarities list in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    seen_chunks = set()
    unique_similarities = []
    # We iterate over the similarities list and add the most relevant chunks to the unique_similarities list
    for chunk, similarity, source in similarities:
        if chunk not in seen_chunks:
            # If the chunk is unique, then we add it to the unique_similarities list
            seen_chunks.add(chunk)
            unique_similarities.append((chunk, similarity, source))
            if len(unique_similarities) == top_k:
                break
    return unique_similarities

def generate_prompt_with_context(relevant_chunks, query):
    """
    Purpose: Generate a prompt that includes the context of the most relevant chunks and the user's query.
    Input: relevant_chunks - the most relevant chunks to the user's query
    Input: query - the user's query
    """
    # Here, we are creating a context that includes the most relevant chunks to the user's query
    context = "Based on the following information: "
    # Here we iterate over the relevant chunks and add them to the context
    for chunk, similarity, source in relevant_chunks:
        context += f"\n- [Source: {source}]: {chunk}"
    # Here we add the user's query to the prompt 
    prompt = f"{context}\n\n{query}"
    return prompt

def generate_text_with_gpt35(prompt, max_tokens=3100, temperature=0.7):
    """
    Purpose: Generate text using the GPT-3.5 model with the specified prompt.
    Input: prompt - the prompt for the model
    Input: max_tokens - the maximum number of tokens to generate
    Input: temperature - controls the randomness of the output, higher values lead to more varied outputs
    """
    response = openai.ChatCompletion.create(
        # Here we use the GPT-3.5-turbo model to generate the text
        model="gpt-3.5-turbo",
        messages=[
            # We make sure to set the role of the system as an expert on the Duke Artificial Intelligence Master of Engineering Program
            {"role": "system", "content": "You are an expert on the Duke Artificial Intelligence Master of Engineering Program"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature, 
        n=1,
        stop=None
    )
    return response.choices[0].message['content'].strip()

def process_questions(input_file, output_file):
    """
    Purpose: Process the questions in the input file and write the answers to the output file.
    Input: input_file - the path to the input file containing the questions.
    Input: output_file - the path to the output file to write the answers.
    """
    # Here we open a new CSV file to write the answers to the questions
    with open(input_file, newline='', encoding='utf-8') as infile, \
         open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        # We use the csv reader to read the questions from the input file
        reader = csv.DictReader(infile)
        # We use the csv writer to write the answers to the output file
        # We define the field names for the CSV file
        writer = csv.DictWriter(outfile, fieldnames=['Question', 'Context', 'Answer'])
        writer.writeheader()

        # For each row in the input file, we process the question and write the answer to the output file
        for row in reader:
            # Here we clean the question by getting rid of the numbering
            clean_question = re.sub(r'^\d+\.\s+', '', row['Question'])
            # Here we print the question that is being processed to make sure that the code is running
            print(f"Processing question: '{clean_question}'")
            # Here we use the find_most_relevant_chunks function to find the most relevant chunks to the user's query
            relevant_chunks = find_most_relevant_chunks(clean_question)
            # If there are relevant chunks, we generate a prompt with the context of the most relevant chunks and the user's query
            if relevant_chunks:
                context = "\n".join(f"{chunk}" for chunk, _, source in relevant_chunks)  # also removed source prefix here
            else:
                context = "No relevant context found."
            # We use the generate_prompt_with_context function to generate a prompt that includes the context of the most relevant chunks and the user's query
            prompt = generate_prompt_with_context(relevant_chunks, clean_question)
            # We use the generate_text_with_gpt35 function to generate text using the GPT-3.5 model with the specified prompt
            answer = generate_text_with_gpt35(prompt)
            # We write the question, context, and answer to the output file
            writer.writerow({'Question': clean_question, 'Context': context, 'Answer': answer})
            # We print the answer to the question to make sure that the code is running
            print("Finished processing and writing to CSV.\n")

# if __name__ == "__main__":
#     process_questions('/content/rephrased_questions.csv', '/content/qaduke5.csv')
