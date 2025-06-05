# src/youtube_fetcher.py

from googleapiclient.discovery import build
import time

def initialize_youtube_client(api_key):
    """
    Initializes and returns the YouTube Data API client.
    Args:
        api_key (str): Your YouTube Data API v3 key.
    Returns:
        googleapiclient.discovery.Resource: The YouTube API service object.
    Raises:
        ConnectionError: If failed to initialize the API client.
    """
    try:
        youtube_client = build('youtube', 'v3', developerKey=api_key)
        print("YouTube API client initialized.")
        return youtube_client
    except Exception as e:
        raise ConnectionError(f"Failed to initialize YouTube API client: {e}")

def _fetch_comments_page(youtube_client, video_id, max_results, page_token):
    """
    Helper function: Fetches a single page of top-level comments for a given video ID.
    """
    request = youtube_client.commentThreads().list(
        part='snippet',
        videoId=video_id,
        textFormat='plainText',
        maxResults=max_results,
        pageToken=page_token
    )
    response = request.execute()
    return response

def get_video_comments(youtube_client, video_id, max_comments_to_fetch=2000):
    """
    Fetches top-level comments for a given YouTube video.

    Args:
        youtube_client (googleapiclient.discovery.Resource): The initialized YouTube API service object.
        video_id (str): The ID of the YouTube video.
        max_comments_to_fetch (int): The maximum number of comments to retrieve.

    Returns:
        list: A list of dictionaries, each representing a comment.
    """
    print(f"Fetching comments for video ID: {video_id}")
    print(f"Targeting up to {max_comments_to_fetch} comments.")

    all_comments_data = []
    next_page_token = None
    comments_fetched_count = 0
    page_num = 1

    while True:
        try:
            results_per_page = min(100, max_comments_to_fetch - comments_fetched_count)
            if results_per_page <= 0:
                break

            response = _fetch_comments_page(
                youtube_client,
                video_id,
                max_results=results_per_page,
                page_token=next_page_token
            )

            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']
                comment_info = {
                    'comment_id': item['id'],
                    'author_display_name': snippet['authorDisplayName'],
                    'author_channel_id': snippet.get('authorChannelId', {}).get('value', None),
                    'text_display': snippet['textDisplay'],
                    'text_original': snippet['textOriginal'],
                    'like_count': snippet['likeCount'],
                    'published_at': snippet['publishedAt'],
                    'updated_at': snippet['updatedAt']
                }
                all_comments_data.append(comment_info)
                comments_fetched_count += 1

                if comments_fetched_count >= max_comments_to_fetch:
                    break

            next_page_token = response.get('nextPageToken')

            if not next_page_token or comments_fetched_count >= max_comments_to_fetch:
                break

            print(f"Fetched {comments_fetched_count} comments (Page {page_num}). Fetching next page...")
            page_num += 1
            # time.sleep(0.1) # Optional: Add a small delay

        except Exception as e:
            print(f"An error occurred while fetching comments: {e}")
            print("This might be due to an invalid video ID, disabled comments, or API quota limits.")
            break

    print(f"\nFinished fetching. Total comments collected: {comments_fetched_count}")
    return all_comments_data



