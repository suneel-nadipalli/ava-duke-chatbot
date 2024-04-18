from scripts.finetuning import train_model

from scripts.prediction import *

from scripts.api_inference import *

from scripts.gist_chunks_to_mongodb import *

from scripts.increase_dataset_size import *

from scripts.question_answer_pairs_dataset import *

from scripts.create_hf_dataset import *

from scripts.evaluation import *

folder_path = "./data/raw"

input_questions_path = './data/processed/questions.csv'

rephrased_questions_path = './data/processed/rephrased_questions.csv'  

output_qa_pairs_path = './data/processed/qac_pairs_700_chunk_size_150.csv'

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

DATASET_NAME = "suneeln-duke/duke_qac_v3"

PEFT_MODEL = "suneeln-duke/dukebot-qac-v1"

MODEL_ID = "suneeln-duke/dukebot-qac-v1-merged"

# login into to HuggingFace account

login()

print("Logged in!")

# Connect to MongoDB Database

db = get_database()

collection = db['Duke5']

# Process the text files in the specified folder and store the text and its embedding in the database.

process_text_files(folder_path, collection)

print("Text Files Processed!")

# Read the questions from the input CSV file

original_questions = read_questions_from_csv(input_questions_path)

print("Questions Read!")

# Increase the dataset size by rephrasing the questions

rephrased_questions = rephrase_questions(original_questions, 700) 

print("Questions Rephrased!")

# Save the rephrased questions to a CSV file

save_questions_to_csv(rephrased_questions, rephrased_questions_path)

print("Rephrased Questions Saved!")

# Create question-answer pairs from the rephrased questions

process_questions(rephrased_questions_path, output_qa_pairs_path)

print("Question-Answer Pairs Created!")

# Create a HuggingFace dataset from the question-answer pairs

create_hf_ds(output_qa_pairs_path, DATASET_NAME)

print("HuggingFace Dataset Created!")

# Train the model on the dataset

model, tokenizer, gen_config, bnb_config = train_model(MODEL_NAME, DATASET_NAME)

print("Done Training!")

# Deploy the model

deploy_model(model, bnb_config, MODEL_NAME, PEFT_MODEL)

print("Model Deployed!")

# Evaluate the model

eval()

print("Evaluation Done!")

# handler = prep_handler(MODEL_ID)

# print("Handler Ready!")

# data_point, dataset = load_data_point(DATASET_NAME, None)

# print("Data Loaded!")

# payload = {
#     "question": data_point["Question"],
#     "context": data_point["Context"],
#     "max_tokens": 200,
#     "temp": 0.3
# }

# answer = answer_question(payload, handler)

# answer = query(payload)