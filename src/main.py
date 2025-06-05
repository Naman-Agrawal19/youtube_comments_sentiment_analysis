# src/main.py (Functions Only)

import yaml
import pandas as pd
import os
from comment_fetcher import initialize_youtube_client, get_video_comments
from preprocessing import clean_comments
from sentiment_analyzer import initialize_sentiment_pipeline, analyze_sentiment

def load_config(config_path='config/config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        exit()

def main():
    # Load configuration
    config = load_config()

    # --- YouTube Fetcher ---
    youtube_config = config['youtube_api']
    try:
        youtube_client = initialize_youtube_client(api_key=youtube_config['api_key'])
    except ConnectionError as e:
        print(f"Exiting due to YouTube API client error: {e}")
        return

    comments_data = get_video_comments(
        youtube_client=youtube_client,
        video_id=youtube_config['video_id'],
        max_comments_to_fetch=youtube_config['max_comments_to_fetch']
    )

    if not comments_data:
        print("No comments fetched. Exiting.")
        return

    # Convert comments to DataFrame
    comments_df = pd.DataFrame(comments_data)
    print(f"\nDataFrame of fetched comments created. Shape: {comments_df.shape}")
    print(comments_df.head())

    # --- Preprocessing ---
    cleaned_comments_df = clean_comments(
        comments_df,
        text_column='text_original' # Use the original text column for cleaning
    )
    print(f"DataFrame after preprocessing. Shape: {cleaned_comments_df.shape}")
    print(cleaned_comments_df.head())

    if cleaned_comments_df.empty:
        print("No comments remaining after cleaning. Exiting.")
        return

    # Extract cleaned text for sentiment analysis
    texts_to_analyze = cleaned_comments_df['cleaned_comment'].tolist()

    # --- Sentiment Analyzer ---
    sentiment_config = config['sentiment_model']
    try:
        sentiment_pipeline = initialize_sentiment_pipeline(
            model_name=sentiment_config['name'],
            max_length=sentiment_config['max_length'],
            pipeline_batch_size=sentiment_config['pipeline_batch_size']
        )
    except RuntimeError as e:
        print(f"Exiting due to Sentiment Pipeline error: {e}")
        return

    sentiment_results = analyze_sentiment(
        sentiment_pipeline=sentiment_pipeline,
        texts=texts_to_analyze,
        pipeline_batch_size=sentiment_config['pipeline_batch_size']
    )

    # --- Attach results to DataFrame ---
    sentiment_labels = [res['label'] for res in sentiment_results]
    sentiment_scores = [res['score'] for res in sentiment_results]

    # Assign results back to the cleaned DataFrame
    cleaned_comments_df['sentiment_label'] = sentiment_labels
    cleaned_comments_df['sentiment_score'] = sentiment_scores

    print("\n--- Final DataFrame with Sentiment Results ---")
    print(cleaned_comments_df[['text_original', 'cleaned_comment', 'sentiment_label', 'sentiment_score']].head())

    # Optional: Save the final DataFrame to a new CSV
    output_csv_filename = f"youtube_comments_with_sentiment.csv"
    cleaned_comments_df.to_csv(output_csv_filename, index=False)
    print(f"\nResults saved to '{output_csv_filename}'")

if __name__ == "__main__":
    main()