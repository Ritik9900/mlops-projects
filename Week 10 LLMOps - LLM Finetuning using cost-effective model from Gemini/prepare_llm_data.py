# prepare_llm_data.py (for Gemini Conversational JSONL Format)

import pandas as pd
import json
import argparse
from sklearn.model_selection import train_test_split

def create_conversational_jsonl(input_path, train_output_path, test_output_path):
    """
    Converts the Iris CSV data into a conversational JSONL format suitable for
    Gemini fine-tuning. Splits the data into a training set for tuning
    and a test set for validation.

    The output format is:
    {"contents": [
        {"role": "user", "parts": [{"text": "..."}]},
        {"role": "model", "parts": [{"text": "..."}]}
    ]}
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    # Split into training and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['species'])
    print(f"Data split into {len(train_df)} training samples and {len(test_df)} test samples.")

    def format_to_jsonl(dataframe, output_path):
        with open(output_path, 'w') as f:
            for _, row in dataframe.iterrows():
                # User's prompt
                user_content = (
                    "What is the species of the iris flower with "
                    f"sepal length {row['sepal_length']}, "
                    f"sepal width {row['sepal_width']}, "
                    f"petal length {row['petal_length']}, "
                    f"and petal width {row['petal_width']}?"
                )

                # Model's response
                assistant_content = row['species']

                # Correct Gemini JSONL schema
                json_record = {
                    "contents": [
                        {"role": "user", "parts": [{"text": user_content}]},
                        {"role": "model", "parts": [{"text": assistant_content}]}
                    ]
                }

                f.write(json.dumps(json_record) + "\n")

        print(f"Successfully created Gemini conversational JSONL file at: {output_path}")

    # Create training and test JSONL files
    format_to_jsonl(train_df, train_output_path)
    format_to_jsonl(test_df, test_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data in Gemini conversational format for fine-tuning.")
    parser.add_argument("--input", type=str, default="data/iris_prepared.csv", help="Path to the input CSV file.")
    parser.add_argument("--train-output", type=str, default="data/iris_conversational_train.jsonl", help="Path for the training JSONL output.")
    parser.add_argument("--test-output", type=str, default="data/iris_conversational_test.jsonl", help="Path for the test JSONL output.")

    args = parser.parse_args()
    create_conversational_jsonl(args.input, args.train_output, args.test_output)
