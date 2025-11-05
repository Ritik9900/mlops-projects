# add_sensitive_feature.py

import pandas as pd
import numpy as np
import argparse

def add_location_attribute(input_path, output_path):
    """
    Loads a dataset, adds a random binary 'location' column,
    and saves the new dataset.
    """
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    # Generate random binary values (0 or 1) for the new 'location' column
    # This simulates a sensitive attribute.
    np.random.seed(42) # for reproducibility
    df['location'] = np.random.randint(0, 2, df.shape[0])
    
    print("Added random 'location' attribute as a sensitive feature.")
    
    # Reorder columns to have 'location' alongside other features
    cols = df.columns.tolist()
    # Example reordering: move location to be after petal_width
    if 'species' in cols:
        species_index = cols.index('species')
        cols.insert(species_index, cols.pop(cols.index('location')))
        df = df[cols]
    
    print(f"Saving new dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a sensitive 'location' attribute to the Iris dataset.")
    parser.add_argument("--input", type=str, default="data/iris_prepared.csv", help="Path to the clean input CSV file.")
    parser.add_argument("--output", type=str, default="data/iris_with_location.csv", help="Path to save the output CSV file with the location attribute.")
    
    args = parser.parse_args()
    
    add_location_attribute(args.input, args.output)