# poison_data.py

import pandas as pd
import numpy as np
import argparse
import os

def poison_data(input_path, output_path, poison_percentage):
    """
    Loads a clean dataset, poisons a specified percentage of its feature rows
    with random numbers, and saves the poisoned dataset.
    """
    print(f"Loading clean data from: {input_path}")
    df = pd.read_csv(input_path)

    if poison_percentage == 0:
        print("Poison percentage is 0. Saving the original file.")
        df.to_csv(output_path, index=False)
        return

    # Identify the feature columns to be poisoned
    feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    # Ensure all feature columns exist in the DataFrame
    for col in feature_columns:
        if col not in df.columns:
            raise ValueError(f"Feature column '{col}' not found in the input data.")
            
    # Calculate the number of rows to poison
    n_rows_to_poison = int(len(df) * (poison_percentage / 100.0))
    if n_rows_to_poison == 0 and poison_percentage > 0:
        print(f"Warning: Poison percentage {poison_percentage}% is too small to affect any rows.")
        df.to_csv(output_path, index=False)
        return
        
    print(f"Poisoning {n_rows_to_poison} out of {len(df)} rows ({poison_percentage}%)...")

    # Randomly select the indices of the rows to poison
    poison_indices = np.random.choice(df.index, n_rows_to_poison, replace=False)
    
    # Create a copy to modify
    poisoned_df = df.copy()

    # Determine a reasonable range for the random noise based on the data's overall min/max
    min_val = poisoned_df[feature_columns].min().min()
    max_val = poisoned_df[feature_columns].max().max()
    print(f"Generating random noise in the range [{min_val:.2f}, {max_val:.2f}]")

    # Replace the values in the selected rows and feature columns with random noise
    for col in feature_columns:
        random_noise = np.random.uniform(min_val, max_val, n_rows_to_poison)
        poisoned_df.loc[poison_indices, col] = random_noise
        
    # Save the poisoned dataset
    print(f"Saving poisoned data to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    poisoned_df.to_csv(output_path, index=False)
    print("Data poisoning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poison a dataset with random noise.")
    parser.add_argument("--input", type=str, required=True, help="Path to the clean input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the poisoned output CSV file.")
    parser.add_argument("--percentage", type=int, required=True, choices=[0, 5, 10, 50], help="Percentage of rows to poison (0, 5, 10, or 50).")
    
    args = parser.parse_args()
    
    poison_data(args.input, args.output, args.percentage)