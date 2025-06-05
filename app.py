import streamlit as st

# --- Streamlit App Configuration (MUST BE FIRST) ---
st.set_page_config(layout="wide", page_title="YouTube Comment Sentiment Analyzer", page_icon="🎥")

import pandas as pd
import re
from urllib.parse import urlparse, parse_qs
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import yaml
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure the src directory is in the Python path
import sys
sys.path.insert(0, './src')

from comment_fetcher import initialize_youtube_client, get_video_comments
from preprocessing import clean_comments
from sentiment_analyzer import initialize_sentiment_pipeline, analyze_sentiment

# --- Custom CSS for enhanced styling ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 0;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card.positive {
        background: linear-gradient(90deg, rgba(0, 236, 167, 1.000) 0.000%, rgba(0, 236, 167, 1.000) 12.500%, rgba(0, 218, 155, 1.000) 12.500%, rgba(0, 218, 155, 1.000) 25.000%, rgba(0, 196, 143, 1.000) 25.000%, rgba(0, 196, 143, 1.000) 37.500%, rgba(0, 171, 131, 1.000) 37.500%, rgba(0, 171, 131, 1.000) 50.000%, rgba(0, 144, 118, 1.000) 50.000%, rgba(0, 144, 118, 1.000) 62.500%, rgba(0, 116, 106, 1.000) 62.500%, rgba(0, 116, 106, 1.000) 75.000%, rgba(0, 88, 93, 1.000) 75.000%, rgba(0, 88, 93, 1.000) 87.500%, rgba(0, 61, 81, 1.000) 87.500% 100.000%)
    }
    
    .metric-card.neutral {
        background: linear-gradient(90deg, rgba(77, 51, 90, 1.000) 0.000%, rgba(77, 51, 90, 1.000) 7.692%, rgba(81, 52, 88, 1.000) 7.692%, rgba(81, 52, 88, 1.000) 15.385%, rgba(87, 55, 89, 1.000) 15.385%, rgba(87, 55, 89, 1.000) 23.077%, rgba(94, 61, 92, 1.000) 23.077%, rgba(94, 61, 92, 1.000) 30.769%, rgba(102, 67, 97, 1.000) 30.769%, rgba(102, 67, 97, 1.000) 38.462%, rgba(109, 76, 105, 1.000) 38.462%, rgba(109, 76, 105, 1.000) 46.154%, rgba(117, 86, 114, 1.000) 46.154%, rgba(117, 86, 114, 1.000) 53.846%, rgba(124, 96, 125, 1.000) 53.846%, rgba(124, 96, 125, 1.000) 61.538%, rgba(129, 107, 138, 1.000) 61.538%, rgba(129, 107, 138, 1.000) 69.231%, rgba(133, 118, 151, 1.000) 69.231%, rgba(133, 118, 151, 1.000) 76.923%, rgba(136, 129, 165, 1.000) 76.923%, rgba(136, 129, 165, 1.000) 84.615%, rgba(136, 139, 179, 1.000) 84.615%, rgba(136, 139, 179, 1.000) 92.308%, rgba(134, 149, 193, 1.000) 92.308% 100.000%)
    }

    .metric-card.negative {
        background: linear-gradient(90deg, rgba(255, 217, 160, 1.000) 0.000%, rgba(255, 217, 160, 1.000) 7.692%, rgba(255, 210, 141, 1.000) 7.692%, rgba(255, 210, 141, 1.000) 15.385%, rgba(255, 201, 119, 1.000) 15.385%, rgba(255, 201, 119, 1.000) 23.077%, rgba(255, 190, 97, 1.000) 23.077%, rgba(255, 190, 97, 1.000) 30.769%, rgba(255, 177, 74, 1.000) 30.769%, rgba(255, 177, 74, 1.000) 38.462%, rgba(255, 163, 51, 1.000) 38.462%, rgba(255, 163, 51, 1.000) 46.154%, rgba(253, 147, 29, 1.000) 46.154%, rgba(253, 147, 29, 1.000) 53.846%, rgba(247, 130, 8, 1.000) 53.846%, rgba(247, 130, 8, 1.000) 61.538%, rgba(240, 112, 0, 1.000) 61.538%, rgba(240, 112, 0, 1.000) 69.231%, rgba(232, 94, 0, 1.000) 69.231%, rgba(232, 94, 0, 1.000) 76.923%, rgba(222, 76, 0, 1.000) 76.923%, rgba(222, 76, 0, 1.000) 84.615%, rgba(211, 58, 0, 1.000) 84.615%, rgba(211, 58, 0, 1.000) 92.308%, rgba(200, 40, 0, 1.000) 92.308% 100.000%)
    }

         
         
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ffffff;
    }
    
    /* Success message styling */
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 500;
    }
    
    /* DataFrames styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration Loading ---
# In a real Streamlit app, you might get the API key from st.secrets
# For demonstration, we'll ask the user to input it or use a placeholder.
# You could also load a config.yaml if it's accessible in the deployment environment.

# Function to load config (adapted for Streamlit, can be replaced by st.secrets)
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

# Load initial configuration
app_config = load_config()

# --- Helper Functions ---

def get_video_id(url):
    """Extracts video ID from a YouTube URL."""
    if "youtube.com/watch" in url:
        query = urlparse(url).query
        video_id = parse_qs(query).get('v')
        if video_id:
            return video_id[0]
    elif "youtu.be/" in url:
        return urlparse(url).path[1:]
    return None

# --- Cache the sentiment pipeline to avoid reloading on every rerun ---
@st.cache_resource
def get_sentiment_pipeline(model_name, max_length, pipeline_batch_size):
    """Initializes and caches the sentiment analysis pipeline."""
    try:
        pipeline = initialize_sentiment_pipeline(
            model_name=model_name,
            max_length=max_length,
            pipeline_batch_size=pipeline_batch_size
        )
        return pipeline
    except RuntimeError as e:
        st.error(f"Error initializing sentiment model: {e}. Please check your internet connection or model name.")
        return None

# --- Streamlit App Layout ---

# Enhanced header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🎥 YouTube Comment Sentiment Analyzer</h1>
    <p class="main-subtitle">Discover the emotional pulse of any YouTube video comments through sentiment analysis</p>
</div>
""", unsafe_allow_html=True)

# Input for YouTube API Key with enhanced styling
with st.sidebar:
    st.markdown("### 🔐 Configuration")
    youtube_api_key = st.text_input(
        "YouTube Data API v3 Key",
        value=app_config['youtube_api']['api_key'],
        type="password",
        help="🔑 Get your API key from Google Cloud Console with YouTube Data API v3 enabled"
    )
    
    if youtube_api_key == 'YOUR_YOUTUBE_API_KEY_HERE' or not youtube_api_key:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    padding: 1rem; border-radius: 10px; color: white; text-align: center;">
            ⚠️ Please enter a valid API key to proceed
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    st.markdown("### ⚙️ Settings")
    max_comments = st.slider(
        "Maximum Comments to Analyze",
        min_value=50,
        max_value=2000,
        value=app_config['youtube_api']['max_comments_to_fetch'],
        step=50,
        help="📊 More comments = better analysis but longer processing time"
    )

# Main input section
st.markdown('<h2 class="section-header">🎯 Video Analysis</h2>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    youtube_url = st.text_input(
        "Enter YouTube Video URL:",
        value="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        placeholder="https://www.youtube.com/watch?v=..."
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("🚀 Analyze Comments", use_container_width=True)

if analyze_button:
    if not youtube_url:
        st.error("Please enter a YouTube video URL.")
    else:
        video_id = get_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL. Please enter a valid video link.")
        else:
            # Progress container
            progress_container = st.container()
            with progress_container:
                st.markdown(f"""
                <div class="info-box">
                    🎬 Analyzing video: <code>{video_id}</code><br>
                    📊 Processing up to {max_comments} comments...
                </div>
                """, unsafe_allow_html=True)

            # Initialize YouTube client
            try:
                youtube_client = initialize_youtube_client(api_key=youtube_api_key)
            except ConnectionError as e:
                st.error(f"Failed to initialize YouTube API client: {e}. Please check your API key and network.")
                st.stop()

            # Fetch comments with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Fetching comments...")
            progress_bar.progress(20)
            
            comments_data = get_video_comments(
                youtube_client=youtube_client,
                video_id=video_id,
                max_comments_to_fetch=max_comments
            )

            if not comments_data:
                st.warning("No comments fetched or comments are disabled for this video. Please try another URL.")
                st.stop()

            comments_df = pd.DataFrame(comments_data)
            
            status_text.text(f"✅ Fetched {len(comments_df)} comments successfully!")
            progress_bar.progress(40)

            # Preprocess comments
            status_text.text("🧹 Cleaning and preprocessing comments...")
            progress_bar.progress(60)
            
            cleaned_comments_df = clean_comments(comments_df, text_column='text_original')

            if cleaned_comments_df.empty:
                st.warning("No comments remaining after cleaning. Cannot perform sentiment analysis.")
                st.stop()

            texts_to_analyze = cleaned_comments_df['cleaned_comment'].tolist()

            # Initialize sentiment pipeline
            sentiment_pipeline = get_sentiment_pipeline(
                model_name=app_config['sentiment_model']['name'],
                max_length=app_config['sentiment_model']['max_length'],
                pipeline_batch_size=app_config['sentiment_model']['pipeline_batch_size']
            )

            if not sentiment_pipeline:
                st.stop()

            # Perform sentiment analysis
            status_text.text("🤖 Analyzing sentiment....")
            progress_bar.progress(80)
            
            sentiment_results = analyze_sentiment(
                sentiment_pipeline=sentiment_pipeline,
                texts=texts_to_analyze,
                pipeline_batch_size=app_config['sentiment_model']['pipeline_batch_size']
            )

            # Attach results to DataFrame
            sentiment_labels = [res['label'] for res in sentiment_results]
            sentiment_scores = [res['score'] for res in sentiment_results]

            label_map = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
            sentiment_labels = [label_map.get(label, label.capitalize()) for label in sentiment_labels]
            cleaned_comments_df['sentiment_label'] = sentiment_labels
            cleaned_comments_df['sentiment_score'] = sentiment_scores

            status_text.text("🎉 Analysis complete!")
            progress_bar.progress(100)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()

            st.markdown("""
            <div class="success-box">
                🎊 Analysis Complete! Scroll down to explore the results
            </div>
            """, unsafe_allow_html=True)

            # Results section
            st.markdown('<h2 class="section-header">📊 Analysis Results</h2>', unsafe_allow_html=True)

            # 1. Enhanced sentiment distribution with custom cards
            st.markdown('<h3 class="section-header">💭 Sentiment Distribution</h3>', unsafe_allow_html=True)
            sentiment_counts = cleaned_comments_df['sentiment_label'].value_counts(normalize=True) * 100

            col1, col2, col3 = st.columns(3)
            
            with col1:
                positive_pct = sentiment_counts.get('Positive', 0)
                st.markdown(f"""
                <div class="metric-card positive">
                    <div class="metric-value">😊 {positive_pct:.1f}%</div>
                    <div class="metric-label">Positive Comments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                neutral_pct = sentiment_counts.get('Neutral', 0)
                st.markdown(f"""
                <div class="metric-card neutral">
                    <div class="metric-value">😐 {neutral_pct:.1f}%</div>
                    <div class="metric-label">Neutral Comments</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                negative_pct = sentiment_counts.get('Negative', 0)
                st.markdown(f"""
                <div class="metric-card negative">
                    <div class="metric-value">😔 {negative_pct:.1f}%</div>
                    <div class="metric-label">Negative Comments</div>
                </div>
                """, unsafe_allow_html=True)

            # Enhanced pie chart with Plotly
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Define colors: Positive - #00D4AA, Neutral - #A839F1, Negative - #E2470A
            sentiment_color_map = {
                'Positive': '#00D4AA',
                'Neutral': '#A839F1',
                'Negative': '#E2470A'
            }
            # Ensure colors are in the same order as sentiment_counts.index
            colors = [sentiment_color_map.get(label, '#CCCCCC') for label in sentiment_counts.index]
            fig_pie = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,  # Donut chart
                marker_colors=colors,
                textinfo='label+percent',
                textfont_size=14,
                hovertemplate='<b>%{label}</b><br>%{percent}<br>Count: %{value:.0f}<extra></extra>',
            )])
            
            fig_pie.update_layout(
                title={
                    'text': "Overall Sentiment Distribution",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20, 'family': 'Inter'}
                },
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                margin=dict(t=60, b=60, l=60, r=60),
                height=500
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 2. Enhanced Word Cloud
            st.markdown('<h3 class="section-header">☁️ Word Cloud</h3>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            all_words = ' '.join(cleaned_comments_df['cleaned_comment'].tolist())
            stopwords = set(STOPWORDS)
            stopwords.update(["youtube", "video", "comment", "like", "channel", "comments", "watch", "get", "go", "dont", "do", "this", "that", "it", "is", "a", "and", "the", "to", "of", "in", "for", "on", "with", "as", "at", "by", "from", "up", "out", "if", "then", "so", "what", "when", "where", "why", "how", "you", "me", "he", "she", "we", "they", "i", "my", "your", "his", "her", "our", "their", "them", "us", "him", "her", "its", "also", "just", "can", "will", "would", "could", "should", "has", "have", "had", "be", "been", "being", "am", "are", "was", "were", "is", "not", "no", "yes", "but", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

            if all_words:
                # Use a modern color palette for wordcloud
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    stopwords=stopwords,
                    min_font_size=12,
                    max_words=100,
                    colormap='viridis',  # Modern colormap
                    relative_scaling=0.5,
                    collocations=False
                ).generate(all_words)

                fig2, ax2 = plt.subplots(figsize=(15, 8))
                ax2.imshow(wordcloud, interpolation='bilinear')
                ax2.axis('off')
                ax2.set_title('Most Frequently Used Words', fontsize=20, fontweight='bold', pad=20)
                plt.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("💭 Not enough words to generate a meaningful word cloud.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # 3. Enhanced Sentiment Over Time with Plotly
            st.markdown('<h3 class="section-header">📈 Sentiment Trends Over Time</h3>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            # Convert 'published_at' to datetime objects
            cleaned_comments_df['published_at'] = pd.to_datetime(cleaned_comments_df['published_at'])
            cleaned_comments_df = cleaned_comments_df.sort_values(by='published_at')
            cleaned_comments_df['date'] = cleaned_comments_df['published_at'].dt.date

            sentiment_over_time = cleaned_comments_df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0)
            sentiment_over_time = sentiment_over_time.reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)

            if not sentiment_over_time.empty:
                fig_time = go.Figure()
                
                colors_time = {'Positive': '#00D4AA', 'Neutral': '#FFB800', 'Negative': '#FF6B6B'}
                
                for sentiment in ['Positive', 'Neutral', 'Negative']:
                    if sentiment in sentiment_over_time.columns:
                        fig_time.add_trace(go.Scatter(
                            x=sentiment_over_time.index,
                            y=sentiment_over_time[sentiment],
                            mode='lines+markers',
                            name=sentiment,
                            line=dict(color=colors_time[sentiment], width=3),
                            marker=dict(size=8, symbol='circle')
                        ))
                
                fig_time.update_layout(
                    title={
                        'text': "Comment Sentiment Trends Over Time",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 20, 'family': 'Inter'}
                    },
                    xaxis_title="Date",
                    yaxis_title="Number of Comments",
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=60, b=80, l=60, r=60),
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                
                fig_time.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                fig_time.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
                
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                st.info("📅 Not enough data points to plot sentiment trends over time.")
            
            st.markdown('</div>', unsafe_allow_html=True)

            # 4. Enhanced Latest Comments Table
            st.markdown('<h3 class="section-header">💬 Latest Comments Analysis</h3>', unsafe_allow_html=True)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            latest_comments_df = cleaned_comments_df.sort_values(by='published_at', ascending=False).head(50)

            # Add emoji indicators for sentiment
            def add_sentiment_emoji(sentiment):
                emoji_map = {'Positive': '😊', 'Neutral': '😐', 'Negative': '😔'}
                return f"{emoji_map.get(sentiment, '❓')} {sentiment}"

            display_columns = ['published_at', 'author_display_name', 'text_original', 'sentiment_label', 'sentiment_score']
            display_df = latest_comments_df[display_columns].copy()
            display_df['sentiment_label'] = display_df['sentiment_label'].apply(add_sentiment_emoji)
            display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
            
            display_df = display_df.rename(columns={
                'published_at': '📅 Published At',
                'author_display_name': '👤 Author',
                'text_original': '💭 Comment',
                'sentiment_label': '🎭 Sentiment',
                'sentiment_score': '📊 Confidence'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600,
                column_config={
                    "💭 Comment": st.column_config.TextColumn(
                        width="large",
                        help="Original comment text"
                    ),
                    "📊 Confidence": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=1,
                        format="%.3f"
                    ),
                }
            )
            
            st.markdown('</div>', unsafe_allow_html=True)

            # Final success message
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; 
                        color: white; margin-top: 2rem;">
                <h3 style="margin-bottom: 1rem;">🎉 Analysis Successfully Completed!</h3>
                <p style="margin-bottom: 0; font-size: 1.1rem; opacity: 0.9;">
                    Found insights from <strong>{cleaned_comments_df.shape[0]}</strong> comments • 
                    Processed with <strong>sentiment analysis</strong> 
                </p>
            </div>
            """, unsafe_allow_html=True)