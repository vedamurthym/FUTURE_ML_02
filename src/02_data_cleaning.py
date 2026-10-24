import os
import re
import pandas as pd
import numpy as np

def get_project_root():
    """
    Dynamically finds the project root folder.
    This script is designed to be executed from either the project root (support-ticket-classification/)
    or the src/ directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        return os.path.dirname(current_dir)
    return current_dir

def load_data(filepath):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(filepath)

def drop_missing_targets(df):
    """
    Checks target columns ('Ticket Type' and 'Ticket Priority') and drops
    rows where either target label is missing. Does not fill missing labels.
    """
    orig_rows = len(df)
    # Check for NaN or empty string after stripping
    temp_df = df.copy()
    
    # Strip target columns first to find empty strings
    for col in ['Ticket Type', 'Ticket Priority']:
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].astype(str).str.strip()
            
    # Replace empty strings or string 'nan'/'none' with actual NaN
    temp_df['Ticket Type'] = temp_df['Ticket Type'].replace(['nan', 'none', ''], np.nan)
    temp_df['Ticket Priority'] = temp_df['Ticket Priority'].replace(['nan', 'none', ''], np.nan)
    
    # Drop rows where target columns are null
    clean_df = df.dropna(subset=['Ticket Type', 'Ticket Priority'])
    # Filter out empty string rows as well
    clean_df = clean_df[
        (clean_df['Ticket Type'].astype(str).str.strip() != "") &
        (clean_df['Ticket Priority'].astype(str).str.strip() != "")
    ]
    
    dropped = orig_rows - len(clean_df)
    print(f"Target column validation: Dropped {dropped} rows with missing target labels.")
    return clean_df

def replace_placeholders(df):
    """
    Safely replaces '{product_purchased}' placeholders in the 'Ticket Description'
    using the actual values in the corresponding 'Product Purchased' column.
    """
    df_copy = df.copy()
    
    def resolve_placeholders_row(row):
        desc = row['Ticket Description']
        prod = row['Product Purchased']
        
        if pd.isna(desc):
            return ""
        
        desc = str(desc)
        if pd.isna(prod):
            return desc
            
        prod = str(prod)
        
        # Replace occurrences of placeholders with the product name using regex
        desc = re.sub(r'\{*\s*product_purchased(?:\.name)?\s*\}*', prod, desc, flags=re.IGNORECASE)
        
        return desc

    df_copy['Ticket Description'] = df_copy.apply(resolve_placeholders_row, axis=1)
    
    # Check remaining placeholders
    remaining = df_copy['Ticket Description'].str.contains(r'product_purchased', case=False, regex=True).sum()
    print(f"Placeholder replacement complete. Remaining 'product_purchased' strings in Description: {remaining}")
    return df_copy, remaining

def clean_text_series(series):
    """
    Cleans a text series by performing basic normalization:
    - handles missing text (converts to string or empty)
    - converts to string
    - lowercases text
    - removes unnecessary leading/trailing whitespace
    - normalizes repeated whitespace
    """
    # Replace nulls with empty string
    cleaned = series.fillna("").astype(str)
    # Lowercase
    cleaned = cleaned.str.lower()
    # Strip whitespace
    cleaned = cleaned.str.strip()
    # Normalize whitespaces (replace multiple whitespaces/newlines with a single space)
    cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
    
    return cleaned

def create_combined_text(df):
    """
    Combines Ticket Subject and Ticket Description into a new column 'ticket_text'.
    Maintains clean whitespace and handles empty strings.
    """
    df_copy = df.copy()
    subject_cleaned = df_copy['Ticket Subject']
    desc_cleaned = df_copy['Ticket Description']
    
    # Combine subject and description
    df_copy['ticket_text'] = subject_cleaned + " " + desc_cleaned
    # Clean any resulting multiple spaces and strip
    df_copy['ticket_text'] = df_copy['ticket_text'].str.strip().str.replace(r'\s+', ' ', regex=True)
    
    return df_copy

def clean_target_labels(df):
    """
    Cleans target categorical labels by stripping accidental whitespace.
    Does not modify class meanings.
    """
    df_copy = df.copy()
    df_copy['Ticket Type'] = df_copy['Ticket Type'].astype(str).str.strip()
    df_copy['Ticket Priority'] = df_copy['Ticket Priority'].astype(str).str.strip()
    return df_copy

def parse_datetime_columns(df):
    """
    Converts Date of Purchase, First Response Time, and Time to Resolution
    to appropriate datetime formats safely, preserving missing values as NaT.
    """
    df_copy = df.copy()
    date_cols = ['Date of Purchase', 'First Response Time', 'Time to Resolution']
    
    for col in date_cols:
        if col in df_copy.columns:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            
    return df_copy

def check_data_quality(df):
    """
    Runs quality checks on the cleaned dataset and returns diagnostic reports.
    """
    quality_issues = []
    
    # 1. Duplicates check
    dups = df.duplicated().sum()
    if dups > 0:
        quality_issues.append(f"Cleaned dataset contains {dups} duplicate rows.")
        
    # 2. Missing target checks
    missing_type = df['Ticket Type'].isna().sum()
    missing_priority = df['Ticket Priority'].isna().sum()
    if missing_type > 0 or missing_priority > 0:
        quality_issues.append(f"Missing labels: Ticket Type ({missing_type} nulls), Ticket Priority ({missing_priority} nulls).")
        
    # 3. Missing ticket_text check
    missing_text = df['ticket_text'].isna().sum()
    empty_text = (df['ticket_text'].str.strip() == "").sum()
    if missing_text > 0 or empty_text > 0:
        quality_issues.append(f"Empty/missing ticket text: {missing_text} nulls, {empty_text} blank rows.")
        
    # 4. Remaining placeholders check
    # Check for anything like {placeholder}
    regex_placeholders = df['Ticket Description'].str.contains(r'\{[a-zA-Z_]+\}', regex=True).sum()
    if regex_placeholders > 0:
        quality_issues.append(f"Unresolved placeholders detected: {regex_placeholders} rows still contain bracket placeholders.")
        
    # 5. Unexpected categories
    expected_types = {'Refund request', 'Technical issue', 'Cancellation request', 'Product inquiry', 'Billing inquiry'}
    unexpected_types = set(df['Ticket Type'].unique()) - expected_types
    if unexpected_types:
        quality_issues.append(f"Unexpected values in 'Ticket Type': {unexpected_types}")
        
    # 6. Unexpected priorities
    expected_priorities = {'Low', 'Medium', 'High', 'Critical'}
    unexpected_priorities = set(df['Ticket Priority'].unique()) - expected_priorities
    if unexpected_priorities:
        quality_issues.append(f"Unexpected values in 'Ticket Priority': {unexpected_priorities}")
        
    return quality_issues

def main():
    project_root = get_project_root()
    raw_path = os.path.join(project_root, "data", "raw", "customer_support_tickets.csv")
    processed_path = os.path.join(project_root, "data", "processed", "customer_support_tickets_cleaned.csv")
    
    print(f"Project root identified: {project_root}")
    print(f"Loading raw dataset from: {raw_path}")
    
    try:
        # Load raw dataset
        df_raw = load_data(raw_path)
        orig_rows, orig_cols = df_raw.shape
        
        # Step 2: Make copy
        df = df_raw.copy()
        
        # Step 3: Remove rows if target labels are missing
        df = drop_missing_targets(df)
        
        # Step 6 & 7: Resolve template placeholders
        df, remaining_placeholders = replace_placeholders(df)
        
        # Step 4 & 5: Clean description and subject columns
        print("Cleaning text fields 'Ticket Description' and 'Ticket Subject'...")
        df['Ticket Description'] = clean_text_series(df['Ticket Description'])
        df['Ticket Subject'] = clean_text_series(df['Ticket Subject'])
        
        # Step 8: Create ticket_text combined column
        print("Creating combined feature 'ticket_text'...")
        df = create_combined_text(df)
        
        # Step 9: Clean target categorical labels
        print("Cleaning categorical target labels...")
        df = clean_target_labels(df)
        
        # Step 10: Convert datetime columns safely
        print("Converting datetime columns safely...")
        df = parse_datetime_columns(df)
        
        # Step 13: Check the cleaned dataset
        quality_issues = check_data_quality(df)
        
        # Save output
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Cleaned dataset saved successfully to: {processed_path}")
        
        # Console output summary
        print("\n" + "="*40)
        print("DATA CLEANING COMPLETED")
        print("="*40)
        print(f"Original rows: {orig_rows}")
        print(f"Final rows: {len(df)}")
        print(f"\nOriginal columns: {orig_cols}")
        print(f"Final columns: {len(df.columns)}")
        
        print("\nCategory distribution:")
        print(df['Ticket Type'].value_counts())
        
        print("\nPriority distribution:")
        print(df['Ticket Priority'].value_counts())
        
        print("\nRemaining missing target values:")
        print(f"  Ticket Type: {df['Ticket Type'].isna().sum()}")
        print(f"  Ticket Priority: {df['Ticket Priority'].isna().sum()}")
        
        print(f"\nRemaining placeholders: {remaining_placeholders}")
        
        print(f"\nCleaned dataset:\n  data/processed/customer_support_tickets_cleaned.csv")
        print(f"\nReport:\n  reports/DATA_CLEANING_REPORT.md")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"An error occurred during dataset cleaning: {e}")

if __name__ == "__main__":
    main()
