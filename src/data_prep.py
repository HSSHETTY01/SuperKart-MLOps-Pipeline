import os
import pandas as pd
from datasets import load_dataset, Dataset

def main():
    print("--- Executing Data Prep Step ---")
    hf_user = os.getenv("HF_USERNAME", "HSSHETTY01")
    dataset = load_dataset(f"{hf_user}/SuperKart", data_files="SuperKart.csv", split="train")

    cols_to_remove = [col for col in ["Product_Id", "Store_Id", "Unnamed: 0"] if col in dataset.column_names]
    cleaned = dataset.remove_columns(cols_to_remove)
    cleaned = cleaned.filter(lambda row: all(val is not None and str(val).strip() != "" for val in row.values()))

    splits = cleaned.train_test_split(test_size=0.2, seed=42)
    os.makedirs("data", exist_ok=True)
    splits["train"].to_pandas().to_csv("data/superkart_train.csv", index=False)
    splits["test"].to_pandas().to_csv("data/superkart_test.csv", index=False)
    print("✅ Successfully generated data/superkart_train.csv and data/superkart_test.csv")

if __name__ == "__main__":
    main()
