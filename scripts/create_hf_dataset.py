from datasets import DatasetDict,Dataset

import pandas as pd

from sklearn.model_selection import train_test_split

def create_hf_ds(data_path, hf_ds_name):
    """
    Purpose: Create a Hugging Face dataset from the question-answer pairs.
    Input: data_path - The path to the question-answer pairs.
    Input: hf_ds_name - The name of the Hugging Face dataset.
    """
    df = pd.read_csv(data_path)

    # Split the dataset into training and testing datasets

    train, test = train_test_split(df, test_size=0.2) 

    train_ds = Dataset.from_pandas(train)

    train_ds = train_ds.remove_columns(["__index_level_0__"])

    test_ds = Dataset.from_pandas(test)

    test_ds = test_ds.remove_columns(["__index_level_0__"])

    # Create a Hugging Face dataset from the training and testing datasets

    duke_qac_ds = DatasetDict(
    {
        'train': train_ds,
        'val': test_ds
    })

    # Push the dataset to the Hugging Face model hub

    duke_qac_ds.push_to_hub(hf_ds_name)