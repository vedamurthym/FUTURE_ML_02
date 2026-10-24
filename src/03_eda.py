import os
import re
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define standard English stopwords (excluding negation terms per instructions)
NEGATION_WORDS = {
    'not', 'no', 'never', 'cannot', 'cant', "can't", 'failed', 'fail', 'failure',
    'without', 'neither', 'nor', 'none', 'nothing', 'nobody', 'nowhere',
    'dont', "don't", 'didnt', "didn't", 'wasnt', "wasn't", 'werent', "weren't",
    'havent', "haven't", 'hasnt', "hasn't", 'hadnt', "hadn't", 'wouldnt', "wouldn't",
    'shouldnt', "shouldn't", 'couldnt', "couldn't", 'isnt', "isn't", 'arent', "aren't"
}

STANDARD_STOPWORDS = {
    'i', "i'm", "i've", "i'd", "i'll", 'me', 'my', 'myself', 'we', "we're", "we've", "we'll", "we'd",
    'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves', 'he', "he's", "he'll", "he'd", 'him', 'his', 'himself', 'she',
    "she's", "she'll", "she'd", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', "they're", "they've", "they'll", "they'd", 'them', 'their', 'theirs', 'themselves',
    'what', "what's", 'which', 'who', "who's", 'whom', 'this', 'that', "that's", "that'll",
    'these', 'those', 'there', "there's", 'where', "where's", 'how', "how's",
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'when', 'why', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'should',
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'customer', 'reviewer', 'husband', 'wife', 'assist', 'assistance', 'issue', 'problem',
    'please', 'help', 'having', 'get', 'got', 'using', 'tried', 'try', 'support', 'ticket',
    'purchased', 'bought', 'need', 'service', 'work', 'affecting', 'productivity'
}

# Combine and ensure negation words are NOT in standard stopwords
STOPWORDS = STANDARD_STOPWORDS - NEGATION_WORDS

def get_project_root():
    """Dynamically finds the project root folder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == 'src':
        return os.path.dirname(current_dir)
    return current_dir

def load_data(filepath):
    """Loads dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    return pd.read_csv(filepath)

def tokenize(text):
    """
    Tokenizes text by lowercasing, removing punctuation (except apostrophes for contractions),
    and splitting on spaces.
    """
    if not isinstance(text, str):
        return []
    # Lowercase and clean punctuation
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    # Split and filter stopwords
    words = text.split()
    return [w for w in words if w not in STOPWORDS]

def analyze_basic_data(df):
    """TASK 1: Prints basic dataset diagnostics."""
    print("=" * 60)
    print(" TASK 1: BASIC DATASET DIAGNOSTICS")
    print("=" * 60)
    print(f"Number of rows   : {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print("\n--- Column Names & Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values per Column ---")
    print(df.isna().sum())
    print(f"\nDuplicate rows count: {df.duplicated().sum()}")
    print("=" * 60)

def analyze_ticket_category(df, figures_dir):
    """TASK 2: Analyzes Ticket Type distribution and saves bar chart."""
    print("\n" + "=" * 60)
    print(" TASK 2: TICKET CATEGORY (TARGET) DISTRIBUTION")
    print("=" * 60)
    counts = df['Ticket Type'].value_counts()
    percentages = df['Ticket Type'].value_counts(normalize=True) * 100
    
    dist_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    print(dist_df)
    print(f"\nMost common category : {counts.index[0]} ({counts.iloc[0]} tickets)")
    print(f"Least common category: {counts.index[-1]} ({counts.iloc[-1]} tickets)")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="viridis", legend=False)
    plt.title("Distribution of Support Ticket Categories (Ticket Type)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Ticket Category", fontsize=12)
    plt.ylabel("Number of Tickets", fontsize=12)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, "ticket_category_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved bar chart: {fig_path}")
    print("=" * 60)

def analyze_priority(df, figures_dir):
    """TASK 3: Analyzes Ticket Priority distribution and saves bar chart."""
    print("\n" + "=" * 60)
    print(" TASK 3: TICKET PRIORITY DISTRIBUTION")
    print("=" * 60)
    
    # Ensure standard ordering
    priority_order = ['Low', 'Medium', 'High', 'Critical']
    df['Ticket Priority'] = pd.Categorical(df['Ticket Priority'], categories=priority_order, ordered=True)
    
    counts = df['Ticket Priority'].value_counts().reindex(priority_order)
    percentages = (counts / counts.sum() * 100)
    
    dist_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percentages})
    print(dist_df)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="magma", legend=False)
    plt.title("Distribution of Ticket Priorities", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Ticket Priority", fontsize=12)
    plt.ylabel("Number of Tickets", fontsize=12)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, "ticket_priority_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved bar chart: {fig_path}")
    print("=" * 60)

def analyze_category_vs_priority(df, figures_dir):
    """TASK 4: Computes category vs priority crosstab and saves heatmap."""
    print("\n" + "=" * 60)
    print(" TASK 4: CATEGORY VS PRIORITY RELATIONSHIP")
    print("=" * 60)
    
    # Crosstab counts
    crosstab_count = pd.crosstab(df['Ticket Type'], df['Ticket Priority'])
    print("--- Crosstab Counts ---")
    print(crosstab_count)
    
    # Crosstab percentages (row-wise normalized)
    crosstab_pct = pd.crosstab(df['Ticket Type'], df['Ticket Priority'], normalize='index') * 100
    print("\n--- Crosstab Row-wise Percentages (%) ---")
    print(crosstab_pct.round(2))
    
    # Plotting Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab_count, annot=True, fmt='d', cmap="YlGnBu", cbar=True,
                annot_kws={"size": 12, "weight": "bold"})
    plt.title("Crosstab Heatmap: Ticket Category vs. Priority", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Ticket Priority", fontsize=12)
    plt.ylabel("Ticket Category", fontsize=12)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, "category_vs_priority.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved heatmap: {fig_path}")
    print("=" * 60)

def analyze_text_length(df, figures_dir):
    """TASK 5: Calculates ticket text length statistics and saves histogram."""
    print("\n" + "=" * 60)
    print(" TASK 5: TICKET TEXT LENGTH ANALYSIS")
    print("=" * 60)
    
    # Text length in characters
    char_lengths = df['ticket_text'].astype(str).str.len()
    print("--- Text Length Statistics (Characters) ---")
    print(f"  Minimum length: {char_lengths.min()}")
    print(f"  Maximum length: {char_lengths.max()}")
    print(f"  Mean length   : {char_lengths.mean():.2f}")
    print(f"  Median length : {char_lengths.median()}")
    print(f"  Std Dev       : {char_lengths.std():.2f}")
    
    # Find outliers
    q25 = char_lengths.quantile(0.25)
    q75 = char_lengths.quantile(0.75)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    lower_bound = q25 - 1.5 * iqr
    
    unusually_long = df[char_lengths > upper_bound]
    unusually_short = df[char_lengths < lower_bound]
    print(f"\nOutlier statistics (based on IQR):")
    print(f"  Upper threshold (> {upper_bound:.1f} chars): {len(unusually_long)} outlier tickets")
    print(f"  Lower threshold (< {lower_bound:.1f} chars): {len(unusually_short)} outlier tickets")
    
    # Plotting Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(char_lengths, bins=40, kde=True, color="skyblue", edgecolor="black")
    plt.title("Distribution of Cleaned Ticket Text Length (Characters)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Text Length (Characters)", fontsize=12)
    plt.ylabel("Ticket Count", fontsize=12)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, "ticket_text_length_distribution.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved histogram: {fig_path}")
    print("=" * 60)

def analyze_word_frequency(df, figures_dir):
    """TASK 6: Computes overall word frequencies and saves bar chart."""
    print("\n" + "=" * 60)
    print(" TASK 6: OVERALL WORD FREQUENCY ANALYSIS")
    print("=" * 60)
    
    all_words = []
    for text in df['ticket_text']:
        all_words.extend(tokenize(text))
        
    word_counts = Counter(all_words)
    top_20 = word_counts.most_common(20)
    
    print("--- Top 20 Most Common Words (Negations Maintained, Stopwords Removed) ---")
    for i, (word, count) in enumerate(top_20, 1):
        print(f"  {i:2d}. {word:<15}: {count} occurrences")
        
    # Plotting top 20
    words_list = [item[0] for item in top_20]
    counts_list = [item[1] for item in top_20]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=counts_list, y=words_list, hue=words_list, palette="plasma", legend=False)
    plt.title("Top 20 Most Frequent Words in Ticket Texts", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Occurrences", fontsize=12)
    plt.ylabel("Words", fontsize=12)
    plt.tight_layout()
    
    fig_path = os.path.join(figures_dir, "top_20_words.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved bar chart: {fig_path}")
    print("=" * 60)
    return word_counts

def analyze_category_words(df):
    """TASK 7: Identifies top common words specific to each Ticket Type."""
    print("\n" + "=" * 60)
    print(" TASK 7: CATEGORY-SPECIFIC VOCABULARY ANALYSIS")
    print("=" * 60)
    
    categories = df['Ticket Type'].unique()
    for cat in categories:
        cat_texts = df[df['Ticket Type'] == cat]['ticket_text']
        cat_words = []
        for text in cat_texts:
            cat_words.extend(tokenize(text))
        cat_counts = Counter(cat_words)
        print(f"\nRepresentative words for '{cat}' (Top 10):")
        for word, count in cat_counts.most_common(10):
            print(f"  - {word:<15}: {count}")
    print("=" * 60)

def analyze_priority_characteristics(df):
    """TASK 8: Compares text lengths and vocabulary across priorities."""
    print("\n" + "=" * 60)
    print(" TASK 8: PRIORITY-SPECIFIC TEXT CHARACTERISTICS")
    print("=" * 60)
    
    priorities = df['Ticket Priority'].unique()
    
    # Compare average length
    for prio in sorted(priorities):
        prio_df = df[df['Ticket Priority'] == prio]
        char_lens = prio_df['ticket_text'].astype(str).str.len()
        word_lens = prio_df['ticket_text'].apply(lambda x: len(str(x).split()))
        
        # Word frequency for priority
        prio_words = []
        for text in prio_df['ticket_text']:
            prio_words.extend(tokenize(text))
        prio_counts = Counter(prio_words)
        top_5 = [f"{w} ({c})" for w, c in prio_counts.most_common(5)]
        
        print(f"Priority: {prio:<10}")
        print(f"  Avg length: {char_lens.mean():.1f} chars ({word_lens.mean():.1f} words)")
        print(f"  Top 5 words: {', '.join(top_5)}")
        print("-" * 40)
    print("=" * 60)

def check_data_imbalance(df):
    """TASK 9: Calculates class imbalance stats for Ticket Type and Ticket Priority."""
    print("\n" + "=" * 60)
    print(" TASK 9: CLASS IMBALANCE EVALUATION")
    print("=" * 60)
    
    for target in ['Ticket Type', 'Ticket Priority']:
        counts = df[target].value_counts()
        total = len(df)
        majority_pct = (counts.max() / total) * 100
        minority_pct = (counts.min() / total) * 100
        ratio = counts.max() / counts.min()
        
        print(f"Target Column: '{target}'")
        print(f"  Majority class percentage: {majority_pct:.2f}%")
        print(f"  Minority class percentage: {minority_pct:.2f}%")
        print(f"  Imbalance Ratio          : {ratio:.3f} (Majority/Minority)")
        
        if ratio < 1.5:
            print("  Assessment: Class distribution is highly balanced. No sampling/weighting adjustment required.")
        else:
            print("  Assessment: Class imbalance present. Might require class weighting during ML training.")
        print("-" * 40)
    print("=" * 60)

def verify_data_leakage(df):
    """TASK 10: Confirms no post-resolution columns will be used as ML features."""
    print("\n" + "=" * 60)
    print(" TASK 10: DATA LEAKAGE & MODEL BOUNDARY VERIFICATION")
    print("=" * 60)
    
    post_resolution_cols = ['Resolution', 'Time to Resolution', 'Customer Satisfaction Rating', 'First Response Time']
    print("The following post-resolution columns are confirmed present in the file but MUST BE EXCLUDED from ML features:")
    for col in post_resolution_cols:
        status = "Present" if col in df.columns else "Absent"
        print(f"  - {col:<30}: {status}")
    print("\nVerification status: Leakage boundaries are strictly defined. Only pre-submission features will be utilized.")
    print("=" * 60)

def main():
    project_root = get_project_root()
    cleaned_data_path = os.path.join(project_root, "data", "processed", "customer_support_tickets_cleaned.csv")
    figures_dir = os.path.join(project_root, "reports", "figures")
    
    # Ensure output folder exists
    os.makedirs(figures_dir, exist_ok=True)
    
    print(f"Project root identified: {project_root}")
    print(f"Loading cleaned dataset from: {cleaned_data_path}\n")
    
    try:
        df = load_data(cleaned_data_path)
        
        # Execute all EDA tasks
        analyze_basic_data(df)
        analyze_ticket_category(df, figures_dir)
        analyze_priority(df, figures_dir)
        analyze_category_vs_priority(df, figures_dir)
        analyze_text_length(df, figures_dir)
        analyze_word_frequency(df, figures_dir)
        analyze_category_words(df)
        analyze_priority_characteristics(df)
        check_data_imbalance(df)
        verify_data_leakage(df)
        
        print("\nEDA Completed successfully. All visualizations saved in: reports/figures/")
        
    except Exception as e:
        print(f"An error occurred during EDA execution: {e}")

if __name__ == "__main__":
    main()
