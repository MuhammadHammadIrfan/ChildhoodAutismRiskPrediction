import pandas as pd
import os

def clean_missing_values(input_path, output_path):
    """
    Clean the dataset by filling missing values according to specified rules.
    
    Parameters:
    input_path (str): Path to the raw CSV file
    output_path (str): Path to save the cleaned CSV file
    """
    # Read the raw data
    df = pd.read_csv(input_path)
    
    print("="*60)
    print("DATA CLEANING PROCESS")
    print("="*60)
    print(f"\nOriginal dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display missing values before cleaning
    print("\nMissing values BEFORE cleaning:")
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing values")
    
    print("\n" + "-"*60)
    print("APPLYING CLEANING RULES:")
    print("-"*60)
    
    # 1. Fill missing age values with mode
    if df['age'].isnull().sum() > 0:
        age_mode = df['age'].mode()[0]
        missing_age_count = df['age'].isnull().sum()
        df['age'].fillna(age_mode, inplace=True)
        print(f"✓ Filled {missing_age_count} missing 'age' values with mode: {age_mode}")
    else:
        print("✓ No missing values in 'age' column")
    
    # 2. Fill missing ethnicity values with "unknown"
    if df['ethnicity'].isnull().sum() > 0:
        missing_ethnicity_count = df['ethnicity'].isnull().sum()
        df['ethnicity'].fillna('unknown', inplace=True)
        print(f"✓ Filled {missing_ethnicity_count} missing 'ethnicity' values with 'unknown'")
    else:
        print("✓ No missing values in 'ethnicity' column")
    
    # 3. Standardize relation column: convert "self" to "Self"
    if 'relation' in df.columns:
        # Count occurrences before standardization
        self_lowercase_count = (df['relation'] == 'self').sum()
        if self_lowercase_count > 0:
            df['relation'] = df['relation'].replace('self', 'Self')
            print(f"✓ Standardized {self_lowercase_count} 'self' values to 'Self' in 'relation' column")
        else:
            print("✓ No 'self' values to standardize in 'relation' column")
    
    # 4. Fill missing relation values with "Parent"
    if df['relation'].isnull().sum() > 0:
        missing_relation_count = df['relation'].isnull().sum()
        df['relation'].fillna('Parent', inplace=True)
        print(f"✓ Filled {missing_relation_count} missing 'relation' values with 'Parent'")
    else:
        print("✓ No missing values in 'relation' column")
    
    # Display missing values after cleaning
    print("\n" + "-"*60)
    print("Missing values AFTER cleaning:")
    missing_after = df.isnull().sum()
    missing_cols_after = missing_after[missing_after > 0]
    if len(missing_cols_after) > 0:
        for col, count in missing_cols_after.items():
            print(f"  {col}: {count} missing values")
    else:
        print("  No missing values remaining!")
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Cleaned data saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Display summary statistics
    print("SUMMARY:")
    print(f"  Total rows: {df.shape[0]}")
    print(f"  Total columns: {df.shape[1]}")
    print(f"  Unique values in 'relation' column: {df['relation'].unique()}")
    print(f"  Value counts for 'relation':")
    print(df['relation'].value_counts().to_string())
    
    return df

if __name__ == "__main__":
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_file = os.path.join(project_root, 'data', 'raw', 'autism_screening.csv')
    output_file = os.path.join(project_root, 'data', 'clean', 'autism_screening_cleaned.csv')
    
    # Check if input file exists
    if os.path.exists(input_file):
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Clean the data
        cleaned_df = clean_missing_values(input_file, output_file)
    else:
        print(f"Error: Input file not found at {input_file}")
