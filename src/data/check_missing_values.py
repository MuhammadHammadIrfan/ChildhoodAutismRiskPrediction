import pandas as pd
import os

def check_missing_values(file_path):
    """
    Analyze missing values in the dataset.
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    dict: Dictionary containing missing value information
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    print(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print("\n" + "="*60)
    print("MISSING VALUES ANALYSIS")
    print("="*60 + "\n")
    
    # Calculate missing values for each column
    missing_data = df.isnull().sum()
    
    # Filter columns with missing values
    columns_with_missing = missing_data[missing_data > 0]
    
    if len(columns_with_missing) > 0:
        print(f"Found {len(columns_with_missing)} columns with missing values:\n")
        
        # Create a detailed report
        missing_info = pd.DataFrame({
            'Column': columns_with_missing.index,
            'Missing Count': columns_with_missing.values,
            'Percentage': (columns_with_missing.values / len(df) * 100).round(2)
        })
        
        print(missing_info.to_string(index=False))
        print(f"\n{'='*60}")
        print(f"Total missing values: {missing_data.sum()}")
        print(f"Total cells: {df.shape[0] * df.shape[1]}")
        print(f"Overall missing percentage: {(missing_data.sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%")
        
        return missing_info
    else:
        print("No missing values found in the dataset!")
        return None

if __name__ == "__main__":
    # Define the path to the raw data file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_file = os.path.join(project_root, 'data', 'raw', 'autism_screening.csv')
    
    # Check if file exists
    if os.path.exists(data_file):
        missing_info = check_missing_values(data_file)
    else:
        print(f"Error: File not found at {data_file}")
