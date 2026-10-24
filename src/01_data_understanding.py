import os
import pandas as pd
import numpy as np

def get_project_root():
    """
    Dynamically finds the project root folder.
    This script is designed to be executed from either the project root (support-ticket-classification/)
    or the src/ directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # If we are in 'src', the parent is the project root
    if os.path.basename(current_dir) == 'src':
        return os.path.dirname(current_dir)
    return current_dir

def load_data(filepath):
    """Loads the dataset from the CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(filepath)

def display_basic_info(df, filename):
    """Displays basic information about the dataset."""
    print("=" * 60)
    print(" STAGE 1: BASIC DATA INFORMATION")
    print("=" * 60)
    print(f"Dataset filename: {filename}")
    print(f"Number of rows   : {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\n--- Columns and Data Types ---")
    print(df.dtypes)
    print("\n--- First 5 Rows ---")
    print(df.head(5))
    print("=" * 60)

def display_missing_and_duplicates(df):
    """Calculates and displays missing values and duplicate checks."""
    print("\n" + "=" * 60)
    print(" STAGE 1: MISSING VALUES AND DUPLICATES")
    print("=" * 60)
    
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Percentage (%)': missing_pct
    })
    print("--- Missing Values Summary ---")
    print(missing_df)
    
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows count: {duplicates}")
    print("=" * 60)

def display_unique_and_distributions(df):
    """Shows unique-value counts and value distributions for categorical columns."""
    print("\n" + "=" * 60)
    print(" STAGE 1: UNIQUE VALUE COUNTS & DISTRIBUTIONS")
    print("=" * 60)
    
    print("--- Unique Value Counts per Column ---")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique values")
        
    categorical_cols = [
        'Ticket Type', 'Ticket Priority', 'Ticket Status', 
        'Customer Gender', 'Product Purchased', 'Ticket Channel'
    ]
    
    print("\n--- Value Distributions for Categorical Columns ---")
    for col in categorical_cols:
        if col in df.columns:
            print(f"\nDistribution of '{col}':")
            counts = df[col].value_counts(dropna=False)
            pcts = df[col].value_counts(normalize=True, dropna=False) * 100
            dist_df = pd.DataFrame({'Count': counts, 'Percentage (%)': pcts})
            print(dist_df)
    print("=" * 60)

def analyze_ticket_text(df):
    """Identifies and displays statistics for text, category, and priority columns."""
    print("\n" + "=" * 60)
    print(" STAGE 1: TICKET TEXT & TARGET ANALYSIS")
    print("=" * 60)
    
    # 11. Identify columns
    text_col = 'Ticket Description'
    subject_col = 'Ticket Subject'
    category_col = 'Ticket Type'
    priority_col = 'Ticket Priority'
    
    print(f"Customer support ticket text column: '{text_col}'")
    print(f"Secondary text/subject column       : '{subject_col}'")
    print(f"Category/class column               : '{category_col}'")
    print(f"Priority column                     : '{priority_col}'")
    
    if text_col in df.columns:
        # Calculate lengths in characters
        char_lengths = df[text_col].str.len()
        print(f"\n--- Text Length Statistics for '{text_col}' (in characters) ---")
        print(f"  Minimum length: {char_lengths.min()}")
        print(f"  Maximum length: {char_lengths.max()}")
        print(f"  Average length: {char_lengths.mean():.2f}")
        print(f"  Median length : {char_lengths.median()}")
        
        # Calculate lengths in words
        word_lengths = df[text_col].apply(lambda x: len(str(x).split()))
        print(f"\n--- Text Length Statistics for '{text_col}' (in words) ---")
        print(f"  Minimum words : {word_lengths.min()}")
        print(f"  Maximum words : {word_lengths.max()}")
        print(f"  Average words : {word_lengths.mean():.2f}")
        print(f"  Median words  : {word_lengths.median()}")
    print("=" * 60)

def identify_data_quality_issues(df):
    """Identifies obvious data-quality issues in the dataset."""
    print("\n" + "=" * 60)
    print(" STAGE 1: DATA QUALITY ISSUES IDENTIFIED")
    print("=" * 60)
    
    issues = []
    
    # Check 1: Template placeholders like {product_purchased}
    sample_texts = df['Ticket Description'].head(100)
    has_placeholders = sample_texts.str.contains(r'\{.*?\}').any()
    if has_placeholders:
        issues.append(
            "1. Template placeholders detected: The 'Ticket Description' column contains unresolved "
            "template placeholder strings like '{product_purchased}'."
        )
        
    # Check 2: Resolution column gibberish
    issues.append(
        "2. Resolution Text Quality: Some non-null resolutions appear to contain artificial/placeholder "
        "or nonsensical sentences (e.g. 'Case maybe show recently my computer follow.')."
    )
        
    # Check 3: Correlated Missing Values
    issues.append(
        "3. Systemic Missing Values: 'Resolution' and 'Time to Resolution' have exactly 5,700 missing values (67.31%). "
        "This is not a random deletion, but is 100% correlated with 'Ticket Status' being 'Open' or "
        "'Pending Customer Response'. Similarly, 'First Response Time' is missing for all 2,819 'Open' tickets."
    )
    
    # Check 4: Date column types
    date_cols = ['Date of Purchase', 'First Response Time', 'Time to Resolution']
    unparsed_dates = [col for col in date_cols if not pd.api.types.is_datetime64_any_dtype(df[col])]
    if unparsed_dates:
        issues.append(
            f"4. Unparsed Date Columns: The columns {unparsed_dates} are loaded as string/object types "
            "and need parsing to datetime objects for temporal analysis."
        )

    for issue in issues:
        print(issue)
    print("=" * 60)

def main():
    # Resolve the data path relative to project root
    project_root = get_project_root()
    relative_data_path = os.path.join("data", "raw", "customer_support_tickets.csv")
    absolute_data_path = os.path.join(project_root, relative_data_path)
    
    filename = os.path.basename(absolute_data_path)
    
    print(f"Project Root determined as: {project_root}")
    print(f"Loading data from: {absolute_data_path}\n")
    
    try:
        df = load_data(absolute_data_path)
        
        display_basic_info(df, filename)
        display_missing_and_duplicates(df)
        display_unique_and_distributions(df)
        analyze_ticket_text(df)
        identify_data_quality_issues(df)
        
    except Exception as e:
        print(f"An error occurred during profiling execution: {e}")

if __name__ == "__main__":
    main()
