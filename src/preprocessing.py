# src/preprocessing.py

import re
import pandas as pd

# Define regex patterns as module-level constants
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
NEWLINE_PATTERN = re.compile(r'[\n\r\t]+')
MULTIPLE_SPACE_PATTERN = re.compile(r'\s+')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)
SPECIAL_CHAR_PATTERN = re.compile(r'[^a-zA-Z0-9\s.,!?;\'"\/]+')

def remove_urls(text):
    """Removes URLs from text."""
    return URL_PATTERN.sub('', text)

def remove_newlines(text):
    """Replaces newline characters with spaces."""
    return NEWLINE_PATTERN.sub(' ', text)

def remove_extra_spaces(text):
    """Replaces multiple spaces with a single space."""
    return MULTIPLE_SPACE_PATTERN.sub(' ', text).strip()

def remove_mentions(text):
    """Removes @mentions from text."""
    return MENTION_PATTERN.sub('', text)

def remove_hashtags(text):
    """Removes #hashtags from text."""
    return HASHTAG_PATTERN.sub('', text)

def remove_emojis(text):
    """Removes common emojis from text."""
    return EMOJI_PATTERN.sub('', text)

def remove_special_characters(text):
    """Removes most special characters, keeping alphanumeric and basic punctuation."""
    return SPECIAL_CHAR_PATTERN.sub('', text)

def clean_comments(comments_df, text_column='text_original'):
    """
    Processes a DataFrame of comments to clean the specified text column.

    Args:
        comments_df (pd.DataFrame): DataFrame containing comments.
        text_column (str): The name of the column containing the comment text.

    Returns:
        pd.DataFrame: DataFrame with the cleaned comments.
    """
    print("\nStarting comment preprocessing...")

    # 1. Handle Nulls and Empty Strings
    cleaned_df = comments_df.dropna(subset=[text_column]).copy()
    cleaned_df[text_column] = cleaned_df[text_column].astype(str).str.strip()
    cleaned_df[text_column] = cleaned_df[text_column].str.lower()
    cleaned_df = cleaned_df[cleaned_df[text_column] != ''].copy()
    initial_count = len(comments_df)
    cleaned_count = len(cleaned_df)
    if initial_count > cleaned_count:
        print(f"Removed {initial_count - cleaned_count} null or empty comments.")

    # Apply cleaning functions in a sensible order
    cleaned_df['cleaned_comment'] = cleaned_df[text_column].apply(remove_urls)
    cleaned_df['cleaned_comment'] = cleaned_df['cleaned_comment'].apply(remove_newlines)
    cleaned_df['cleaned_comment'] = cleaned_df['cleaned_comment'].apply(remove_mentions)
    cleaned_df['cleaned_comment'] = cleaned_df['cleaned_comment'].apply(remove_hashtags) # Decide if you want to keep hashtags
    cleaned_df['cleaned_comment'] = cleaned_df['cleaned_comment'].apply(remove_emojis)
    cleaned_df['cleaned_comment'] = cleaned_df['cleaned_comment'].apply(remove_special_characters)
    cleaned_df['cleaned_comment'] = cleaned_df['cleaned_comment'].apply(remove_extra_spaces)

    # Remove any rows that became empty after cleaning
    pre_final_count = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df['cleaned_comment'] != ''].copy()
    if pre_final_count > len(cleaned_df):
        print(f"Removed {pre_final_count - len(cleaned_df)} comments that became empty after cleaning.")

    print("Comment preprocessing complete.")
    return cleaned_df
