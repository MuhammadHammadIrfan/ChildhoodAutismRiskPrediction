import pandas as pd
import os

def remove_relation_and_country_columns(input_path, output_path):
    """
    Remove all columns related to 'relation' and 'country' from the encoded dataset.
    Keep only: A1-A10 scores, age, gender, jaundice, autism, used_app_before, class, and ethnicity columns.
    
    Parameters:
    input_path (str): Path to the encoded CSV file
    output_path (str): Path to save the updated CSV file
    """
    # Read the encoded data
    df = pd.read_csv(input_path)
    
    print("="*60)
    print("REMOVING RELATION AND COUNTRY COLUMNS")
    print("="*60)
    print(f"\nOriginal dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Display all columns before removal
    print("\nColumns BEFORE removal:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Identify columns to remove
    app_used = 'used_app_before'
    country_columns = [col for col in df.columns if col.startswith('country_')]
    relation_columns = [col for col in df.columns if col.startswith('relation_')]
    columns_to_remove = country_columns + relation_columns + [app_used]
    
    print("\n" + "-"*60)
    print("COLUMNS TO REMOVE:")
    print("-"*60)
    print(f"\nCountry columns ({len(country_columns)}):")
    for col in country_columns:
        print(f"  - {col}")
    
    print(f"\nRelation columns ({len(relation_columns)}):")
    for col in relation_columns:
        print(f"  - {col}")
    
    print(f"\nTotal columns to remove: {len(columns_to_remove)}")
    
    # Remove the columns
    df_updated = df.drop(columns=columns_to_remove)
    
    # Display columns after removal
    print("\n" + "-"*60)
    print("COLUMNS AFTER REMOVAL:")
    print("-"*60)
    for i, col in enumerate(df_updated.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n{'='*60}")
    print(f"Updated dataset shape: {df_updated.shape[0]} rows × {df_updated.shape[1]} columns")
    print(f"{'='*60}")
    
    # Save updated data
    df_updated.to_csv(output_path, index=False)
    print(f"\n✓ Updated data saved to: {output_path}")
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"  Original columns: {df.shape[1]}")
    print(f"  Removed columns: {len(columns_to_remove)}")
    print(f"  Remaining columns: {df_updated.shape[1]}")
    print(f"  Total rows: {df_updated.shape[0]}")
    
    print("\nRemaining feature categories:")
    print(f"  - AQ-10 Scores: 10 columns (A1_Score to A10_Score)")
    print(f"  - Demographics: 5 columns (age, gender, jaundice, autism, used_app_before)")
    print(f"  - Ethnicity: {len([c for c in df_updated.columns if c.startswith('ethnicity_')])} columns")
    print(f"  - Target: 1 column (class)")
    print("="*60 + "\n")
    
    return df_updated

if __name__ == "__main__":
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    input_file = os.path.join(project_root, 'data', 'clean', 'autism_screening_encoded.csv')
    output_file = os.path.join(project_root, 'data', 'clean', 'autism_screening_encoded.csv')
    
    # Check if input file exists
    if os.path.exists(input_file):
        # Remove columns
        updated_df = remove_relation_and_country_columns(input_file, output_file)
        print(f"✓ Process completed successfully!")
    else:
        print(f"Error: Input file not found at {input_file}")
