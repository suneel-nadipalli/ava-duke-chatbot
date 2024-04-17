import transformers, torch

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig  
)

import configparser

config = configparser.ConfigParser()


# Read the configuration file

config.read('configs/config.ini')

BITS_AND_BYTES = config['BITS_AND_BYTES']

LORA_CONFIG = config['LORA_CONFIG']

GEN_CONFIG = config['GEN_CONFIG']

TRAINING_ARGS = config['TRAINING_ARGS']

def print_trainable_parameters(model):
    """
    Purpose: Print the number of trainable parameters in the model.
    Input: model - The model to print the number of trainable parameters for.
    """

    trainable_params = 0
    all_param = 0

    # Loop through the parameters in the model and sum the number of trainable parameters

    for _, param in model.named_parameters():
        all_param += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || trainable %: {100* trainable_params/all_param}"
    )


def generate_prompt(data_point):

    """
    Purpose: Generate a prompt from the data point.
    Input: data_point - The data point to generate the prompt from.
    """

    system_message = "Use the provided context followed by a question to answer it."

    # Generate the prompt based on the format
    
    full_prompt = f"""<s>### Instruction:
    {system_message}
    
    ### Context:
    {data_point['Context']}
    
    
    ### Question:
    
    {data_point['Question']}
    
    
    ### Aswer: 
    {data_point['Answer']}
    """
    
    full_prompt = " ".join(full_prompt.split())
    
    return full_prompt

def generate_and_tokenize_prompt(data_point, tokenizer):
    """
    Purpose: Generate and tokenize the prompt from the data point.
    Input: data_point - The data point to generate the prompt from.
    """
    
    # Generate the prompt from the data point
    
    full_prompt = generate_prompt(data_point)
    
    # Tokenize the prompt

    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    
    return tokenized_full_prompt

def init_model_and_tokenizer(MODEL_NAME):
    """
    Purpose: Initialize the model and tokenizer.
    Input: MODEL_NAME - The name of the model to initialize.
    """

    # Initialize the model and tokenizer with Bits and Bytes configuration

    bnb_config = BitsAndBytesConfig(
    load_in_4bit = config.getboolean('BITS_AND_BYTES', 'load_in_4bit'),
    bnb_4bit_quant_type = BITS_AND_BYTES['bnb_4bit_quant_type'],
    bnb_4bit_use_double_quant = config.getboolean('BITS_AND_BYTES', 'bnb_4bit_use_double_quant'),
    bnb_4bit_compute_dtype = torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='auto',
        quantization_config=bnb_config,
        use_cache=False,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, bnb_config

def prepare_model_config(model, tokenizer):
    """
    Purpose: Prepare the model configuration.
    Input: model - The model to prepare the configuration for.
    Input: tokenizer - The tokenizer to prepare the configuration for.
    """
    
    # Enable gradient checkpointing

    model.gradient_checkpointing_enable()
    
    # Prepare the model for kbit training

    model = prepare_model_for_kbit_training(model)

    # Initialize the Lora configuration

    config = LoraConfig(
    r = LORA_CONFIG['r'],
    lora_alpha = LORA_CONFIG['lora_alpha'],
    lora_dropout = LORA_CONFIG['lora_dropout'],
    bias = LORA_CONFIG['bias'],
    task_type = LORA_CONFIG['task_type'],
    )

    # Get the PEFT model

    model = get_peft_model(model, config)
    
    print_trainable_parameters(model)

    # Initialize the generation configuration

    gen_config = model.generation_config
    gen_config.max_new_tokens = GEN_CONFIG['max_new_tokens']
    gen_config.temperature = GEN_CONFIG['temperature']
    gen_config.top_p = GEN_CONFIG['top_p']
    gen_config.num_return_sequences = GEN_CONFIG['num_return_sequences']
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer, gen_config

def prepare_dataset(dataset_name):

    """
    Purpose: Prepare the dataset for training.
    Input: dataset_name - The name of the dataset to prepare.
    """

    # Load the dataset

    dataset = load_dataset(dataset_name)

    # Shuffle and map the dataset to format, generate and tokenize the prompt for each data point

    train_dataset = dataset["train"].shuffle().map(generate_and_tokenize_prompt)
    
    val_dataset = dataset["val"].shuffle().map(generate_and_tokenize_prompt)

    return train_dataset, val_dataset

def train_model(model_name, dataset_name):
    """
    Purpose: Train the model.
    Input: model_name - The name of the model to train.
    Input: dataset_name - The name of the dataset to train on.
    """

    # Initialize the model and tokenizer

    model, tokenizer, bnb_config = init_model_and_tokenizer(model_name)

    # Prepare the model configuration

    model, tokenizer, gen_config = prepare_model_config(model, tokenizer)

    train_dataset, val_dataset = prepare_dataset(dataset_name)

    # Initialize the training arguments

    training_args = transformers.TrainingArguments(
    per_device_train_batch_size = TRAINING_ARGS['per_device_train_batch_size'],
    gradient_accumulation_steps = TRAINING_ARGS['gradient_accumulation_steps'],
    num_train_epochs = TRAINING_ARGS['num_train_epochs'],
    learning_rate = TRAINING_ARGS['learning_rate'],
    fp16 = config.getboolean('TRAINING_ARGS', 'fp16'),
    save_total_limit = TRAINING_ARGS['save_total_limit'],
    logging_steps = TRAINING_ARGS['logging_steps'],
    output_dir = TRAINING_ARGS['output_dir'],
    max_steps = TRAINING_ARGS['max_steps'],
    optim = TRAINING_ARGS['optim'],
    lr_scheduler_type = TRAINING_ARGS['lr_scheduler_type'],
    warmup_ratio = TRAINING_ARGS['warmup_ratio'],
    )

    # Initialize the trainer

    trainer = transformers.Trainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    args = training_args,
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Disable the cache

    model.config_use_cache = False

    # Train the model

    trainer.train()

    return model, tokenizer, gen_config, bnb_config