import pandas as pd
from ucimlrepo import fetch_ucirepo 
import os

def fetch_and_save_data():
    print("Fetching data from UCI repository...")
    
    # 1. Fetch dataset using the ID provided in your snippet
    autism_dataset = fetch_ucirepo(id=419) 
    
    # 2. Extract features and targets
    X = autism_dataset.data.features 
    y = autism_dataset.data.targets 
    
    # 3. Combine them into one DataFrame
    # We concatenate along columns (axis=1) so the target is the last column
    df = pd.concat([X, y], axis=1)
    
    # 4. Create the directory if it doesn't exist
    # This saves it into the 'data/raw/' folder
    save_path = os.path.join('data', 'raw')
    os.makedirs(save_path, exist_ok=True)
    
    # 5. Save to CSV
    csv_filename = os.path.join(save_path, 'autism_screening.csv')
    df.to_csv(csv_filename, index=False)
    
    print(f"Success! Data saved to {csv_filename}")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

if __name__ == "__main__":
    fetch_and_save_data()