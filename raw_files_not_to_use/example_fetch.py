# simple_youtube_request.py

from googleapiclient.discovery import build
import json # To pretty-print the JSON response
import yaml
import os
def load_config(config_path='../config/config.yaml'):
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
config = load_config()
# --- Configuration ---
YOUTUBE_API_KEY = config['youtube_api']['api_key'] # <<< IMPORTANT: Replace with your actual API key
YOUTUBE_VIDEO_ID = config['youtube_api']['video_id'] # <<< Replace with your target video ID (e.g., a short, popular video for quick testing)

# --- 1. Initialize YouTube API Client ---
try:
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    print("YouTube API client initialized.")
except Exception as e:
    print(f"Error initializing YouTube API client: {e}")
    print("Please check your API key and internet connection.")
    exit()

# --- 2. Make a single API request for comment threads ---
# We'll set maxResults to a small number (e.g., 5) to get a quick sample.
# We omit pageToken to get the first page.
try:
    print(f"\nMaking API request for video ID: {YOUTUBE_VIDEO_ID}")
    request = youtube.commentThreads().list(
        part='snippet',
        videoId=YOUTUBE_VIDEO_ID,
        textFormat='plainText',
        maxResults=5 # Requesting a small number of comments for a single page sample
    )

    # The 'request' object itself holds the details of the API call.
    # We can't directly print the 'original request API' in a human-readable URL format easily from this object,
    # because google-api-python-client abstracts that.
    # However, executing it will show you the *result* of that request.

    response = request.execute()

    print("\n--- Raw API Response Sample (JSON) ---")
    print(json.dumps(response, indent=2)) # Pretty-print the JSON response

    print("\n--- Key fields from the first comment (if available) ---")
    if response and response.get('items'):
        first_comment_item = response['items'][0]
        snippet = first_comment_item['snippet']['topLevelComment']['snippet']
        print(f"Comment ID: {first_comment_item['id']}")
        print(f"Author: {snippet['authorDisplayName']}")
        print(f"Original Text: {snippet['textOriginal']}")
        print(f"Like Count: {snippet['likeCount']}")
        print(f"Published At: {snippet['publishedAt']}")
    else:
        print("No comments found in the response, or response is empty.")

    print("\n--- Pagination Info ---")
    if 'nextPageToken' in response:
        print(f"Next Page Token exists: {response['nextPageToken']}")
    else:
        print("No Next Page Token found (likely reached end of comments or very few comments).")
    print(f"Total Results (estimated): {response['pageInfo']['totalResults']}")
    print(f"Results Per Page (in this response): {response['pageInfo']['resultsPerPage']}")


except Exception as e:
    print(f"An error occurred during the API request: {e}")
    print("Common reasons: Invalid video ID, comments disabled, API quota exceeded, or network issue.")