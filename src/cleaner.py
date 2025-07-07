import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load CSV into DataFrame."""
    return pd.read_csv(path, encoding='utf-8')

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename inconsistent or camel-case columns."""
    return df.rename(columns={
        'Id': 'id',
        'News content': 'content',
        'Label': 'label'
    })

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types."""
    df['date'] = pd.to_datetime(df['date'])
    df['label'] = df['label'].str.lower()
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates based on content and title."""
    df = df.drop_duplicates(subset=['content'])
    df = df.drop_duplicates(subset=['title'])
    return df

def remove_outdated_articles(df: pd.DataFrame, date_cutoff: str = '2023-01-01') -> pd.DataFrame:
    """Remove articles before a specific date."""
    df['date'] = pd.to_datetime(df['date'])
    return df[df['date'] >= date_cutoff].copy()

def group_rare_platforms(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """Group infrequent platforms into 'Other'."""
    platform_counts = df['platform'].value_counts()
    df['platform_grouped'] = df['platform'].apply(
        lambda x: x if platform_counts[x] >= threshold else 'Other'
    )
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop irrelevant columns."""
    return df.drop(columns=['id', 'platform'])

def apply_all_cleaning(path: str) -> pd.DataFrame:
    """Full structural cleaning pipeline."""
    df = load_data(path)
    df = rename_columns(df)
    df = fix_dtypes(df)
    df = drop_duplicates(df)
    df = remove_outdated_articles(df)
    df = group_rare_platforms(df)
    df = drop_unused_columns(df)
    return df

if __name__ == "__main__":
    cleaned_df = apply_all_cleaning("data/raw/original_news_data.csv")
    cleaned_df.to_csv("data/processed/cleaned_news.csv", index=False)
    print("Data cleaned and saved.")