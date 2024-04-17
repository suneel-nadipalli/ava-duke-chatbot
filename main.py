from scripts.finetuning import train_model

from scripts.prediction import *

from scripts.api_inference import *

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

DATASET_NAME = "suneeln-duke/duke_qac_v3"

PEFT_MODEL = "suneeln-duke/dukebot-qac-v1"

MODEL_ID = "suneeln-duke/dukebot-qac-v1-merged"

model, tokenizer, gen_config, bnb_config = train_model(MODEL_NAME, DATASET_NAME)

print("Done Training!")

login()

print("Logged in!")

deploy_model(model, bnb_config, MODEL_NAME, PEFT_MODEL)

print("Model Deployed!")

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
