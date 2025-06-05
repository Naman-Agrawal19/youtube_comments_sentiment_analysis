# YouTube Comment Sentiment Analysis

This project provides a complete pipeline for fetching, cleaning, and analyzing the sentiment of YouTube comments, with a focus on Indian and Hinglish content. It leverages modern NLP models and provides a Streamlit-based dashboard for interactive exploration.

---

## Features

- **YouTube Comment Fetching:**  
  Fetches comments from YouTube videos using the YouTube Data API.

- **Preprocessing:**  
  Cleans and normalizes comments, removes stopwords (supports English, Hindi, and Hinglish), and handles emojis, URLs, and special characters.

- **Sentiment Analysis:**  
  Supports multiple sentiment analysis models, including:
  - `ganeshkharad/gk-hinglish-sentiment`
  - `pascalrai/hinglish-twitter-roberta-base-sentiment`
  - VADER and TextBlob for classical sentiment scoring

- **Visualization:**  
  Interactive dashboard built with Streamlit for visualizing sentiment distribution, word clouds, and more.

- **Batch Processing:**  
  Efficient batch inference for large datasets.

---

## Directory Structure

```
prime-gamer-youtube_comments_sentiment_analysis/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── screenshots/                # Screenshots for documentation
├── raw_files_not_to_use/       # EDA and experimental scripts
│   ├── eda.py
│   ├── example_fetch.py
│   └── example_request.jsonc
└── src/                        # Core modules
    ├── comment_fetcher.py
    ├── main.py
    ├── preprocessing.py
    └── sentiment_analyzer.py
```

---

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd prime-gamer-youtube_comments_sentiment_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**
   - Place your YouTube Data API key in `config/config.yaml` or provide it via the Streamlit UI.

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## Screenshots

Below are some screenshots of the dashboard and features:

### Dashboard Overview

![Dashboard Overview](screenshots/dashboard_overview.png)

### Sentiment Distribution

![Sentiment Distribution](screenshots/sentiment_distribution.png)

### Word Cloud

![Word Cloud](screenshots/wordcloud.png)

---

## Customization

- **Models:**  
  You can easily swap out or add new Hugging Face models in `src/sentiment_analyzer.py`.

- **Stopwords:**  
  The stopword lists can be extended for better Hinglish/Indian language support.

- **Preprocessing:**  
  Modify `src/preprocessing.py` to adjust cleaning steps as needed.

---

## Notes

- The `raw_files_not_to_use/` folder contains exploratory and experimental scripts. The main pipeline is in `src/`.
- For best performance on large datasets, use a GPU.
- If you encounter issues with model downloads, ensure your internet connection is active and the Hugging Face `transformers` library is up to date.

---

## License

This project is for educational and research purposes.

---

## Acknowledgements

- Hugging Face for open-source models and pipelines
- Streamlit for rapid dashboard development
- NLTK, spaCy, and TextBlob for NLP utilities

---
