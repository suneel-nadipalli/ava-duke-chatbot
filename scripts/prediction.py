from typing import Any, Dict

import torch, re

import pandas as pd

import transformers

from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

from huggingface_hub import notebook_login

from peft import (
    PeftConfig,
    PeftModel,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from huggingface_hub import notebook_login

import huggingface_hub

import configparser

config = configparser.ConfigParser()

config.read('configs/config.ini')

SECRETS = config['SECRETS']

GEN_CONFIG = config['GEN_CONFIG']

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16

class EndpointHandler:

    """
    Purpose: This class is responsible for handling the endpoint for the model.
    """

    def __init__(self, path=""):
        """
        Purpose: Initialize the class with the model and tokenizer.
        Input: path - The path to the model and tokenizer.
        """

        # Load the model and tokenizer.

        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code = True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            return_dict = True,
            device_map = "auto",
            load_in_8bit = True,
            torch_dtype = dtype,
            trust_remote_code = True,
        )
        
        # Set the generation configuration.

        gen_config = model.generation_config
        gen_config.max_new_tokens = GEN_CONFIG['max_new_tokens']
        gen_config.temperature = GEN_CONFIG['temperature']
        gen_config.top_p = GEN_CONFIG['top_p']
        gen_config.num_return_sequences = GEN_CONFIG['num_return_sequences']
        gen_config.pad_token_id = tokenizer.eos_token_id
        gen_config.eos_token_id = tokenizer.eos_token_id
        
        self.generation_config = gen_config

        # Set the pipeline.
        
        self.pipeline = transformers.pipeline(
            'text-generation', model=model, tokenizer=tokenizer
        )
       
     
    def __call__(self, data: Dict[dict, Any]) -> Dict[str, Any]:
        """
        Purpose: Answer the question based on the context provided.
        Input: data - The data containing the question and context.
        """

        # Extract the question and context from the data.

        question = data.pop("question", data)
        
        context = data.pop("context", None)
        
        temp = data.pop("temp", None)
        
        max_tokens = data.pop("max_tokens", None)
        
        system_message = "Use the provided context followed by a question to answer it."

        # Generate the prompt based on the format

        full_prompt = f"""<s>### Instruction:
        {system_message}

        ### Context:
        {context}


        ### Question:

        {question}


        ### Answer: 
        """

        full_prompt = " ".join(full_prompt.split())

        # Set the generation configuration's temperature and max tokens.
        
        self.generation_config.max_new_tokens = max_tokens
        self.generation_config.temperature = temp
        
        result = self.pipeline(full_prompt, generation_config = self.generation_config)[0]['generated_text']

        # Extract the answer from the result.
               
        match = re.search(r'### Answer:(.*?)###', result, re.DOTALL)
        
        if match:
            result =  match.group(1).strip()
            
        pattern = r"### Answer:(.*)"

        match = re.search(pattern, result)
        
        if match:
            result = match.group(1).strip()      
        
        return result

def login():
    """
    Purpose: Login to the Hugging Face model hub.
    """
    notebook_login()

    huggingface_hub.login(token = SECRETS["hf_token"])

def deploy_model(model, bnb_config, MODEL_NAME, PEFT_MODEL):
    """
    Purpose: Deploy the model to the Hugging Face model hub.
    Input: model - The model to deploy.
    Input: bnb_config - The BitsAndBytes configuration.
    Input: MODEL_NAME - The name of the model.
    Input: PEFT_MODEL - The name of the PEFT model.
    """

    # Push the model to the Hugging Face model hub.
    
    model.push_to_hub(
    PEFT_MODEL,
    use_auth_token=True
    )

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16

    config = PeftConfig.from_pretrained(PEFT_MODEL)

    # Load the model and tokenizer.

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict = True,
        quantization_config = bnb_config,
        device_map = "auto",
        torch_dtype = dtype,
        trust_remote_code = True
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
    tokenizer.pad_token = tokenizer.eos_token

    # Load the PEFT model

    model = PeftModel.from_pretrained(model, PEFT_MODEL)

    # Merge the model and unload it.

    model = model.merge_and_unload()

    # Push the merged model to the Hugging Face model hub.

    model.push_to_hub(
    f"suneeln-duke/{PEFT_MODEL}-merged",
    use_auth_token=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.push_to_hub(
        f"suneeln-duke/{PEFT_MODEL}-merged",
        use_auth_token=True
    )

def prep_handler(MODEL_ID):
    """
    Purpose: Prepare the handler for the model.
    Input: MODEL_ID - The ID of the model.
    """
    my_handler = EndpointHandler(path=MODEL_ID)

    return my_handler

def answer_question(payload, handler):
    """
    Purpose: Answer the question based on the payload and handler.
    Input: payload - The payload containing the question and context.
    Input: handler - The handler for the model.
    """
    return handler(payload)

def load_data_point(DATASET_NAME, dataset=None):
    """
    Purpose: Load a data point from the dataset.
    Input: DATASET_NAME - The name of the dataset.
    Input: dataset - The dataset to load the data point from.
    """
    
    if dataset is None:
        dataset = load_dataset(DATASET_NAME)

    data_point = pd.DataFrame(dataset['val']).sample().to_dict()

    return data_point, dataset