import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from collections import Counter

df1 = pd.read_csv("Reddit_Data.csv") 
df2 = pd.read_csv("Twitter_Data.csv")

df2.dropna(inplace=True)
df2.shape

df1.dropna(inplace=True)
df1.shape

df = pd.concat([pd.DataFrame(df1.values), pd.DataFrame(df2.values)], ignore_index=True,  axis=0)
df.columns = ["comment", "category"]
df['comment'] = df['comment'].astype(str).apply(lambda x: ' '.join(x.split()).lower())
df['comment'] = df['comment'].str.replace(r'\b(jpg|png)\b', '', regex=True).str.strip()
df['len_comment'] = df['comment'].astype(str).apply(lambda x: len(x)) 
df = df[df['len_comment']>2]  # remove comments with less than 3 characters because they are emojis

df.isna().sum()
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.shape
df['has_url'] = df['comment'].str.contains(r'http[s]?://|www\.', regex=True)
df['has_url'].value_counts() # no url in comments
df = df.iloc[:, :2]
df.iloc[:, 1].value_counts().plot(kind='bar')
df['word_count'] = df['comment'].astype(str).apply(lambda x: len(x.split()))


for i in df['category'].unique():
    sns.kdeplot(df.loc[df['category']==i,'word_count'],label=i, fill=True, alpha=0.5)
plt.title('Distribution of Word Counts by Category')
plt.xlabel('Word Count')
plt.ylabel('Density')
plt.xlim(0, 100)
plt.legend()
plt.show()

for i in df['category'].unique():
    sns.boxplot(x='category', y='word_count', data=df.loc[df['category']==i,:])
plt.title('Distribution of Word Counts by Category')
plt.xlabel('Word Count')
plt.ylabel('Density')
plt.legend()
plt.show()

sns.barplot(x='category', y='word_count', data=df, estimator='median')

## seems like distribution of word counts for 1 and -1 is quite similar but for 0 is quite different

# checking for stop words
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords

# as our comments in mostly hinglish and english, we using both stopwords
stop_words = stopwords.words('english') 
print(len(stop_words))


# counting the number of stop words in each comment
df['stop_word_count'] = df['comment'].astype(str).apply(lambda x: len([word for word in x.split() if word in stop_words]))

sample = df.sample(10)

sns.histplot(df['stop_word_count'], kde=True, stat='density')
sns.displot(df['stop_word_count'], kind='hist', kde=True)
plt.title('Distribution of Stop Word Counts')
plt.xlabel('Stop Word Count')
plt.ylabel('Density')
plt.xlim(0, 100)
plt.legend()
plt.show()

def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

all_stop_word = [word for comment in df['comment'] for word in comment.split() if word in stop_words]
most_common_stop = Counter(all_stop_word).most_common(25)
top_25_stop = pd.DataFrame(most_common_stop, columns=['stop_word', 'count'])
sns.barplot(y='stop_word', x='count', data=top_25_stop, palette='viridis')
plt.title('Top 25 Stop Words')
plt.xlabel('Stop Word')
plt.ylabel('Count')
plt.show()

# most frequent words
most_frequent_words = Counter(' '.join(df['comment']).split()).most_common(25)
sns.barplot(hue='word', y='word', x='count', data=pd.DataFrame(most_frequent_words, columns=['word', 'count']), palette='viridis', legend=False)
plt.title('Most Frequent Words')
plt.xlabel('Word')
plt.ylabel('Count')
plt.show()

# checking for punctuation count
df['punctuation_count'] = df['comment'].astype(str).apply(lambda x: len([word for word in x.split() if word in string.punctuation]))
df['punctuation_count'].value_counts().sort_index()
df[df['punctuation_count'] == df['punctuation_count'].unique()[-6]]['comment'].values

# removing punctuation from comments with count greater than 0
df.loc[df['punctuation_count']>0, 'comment'] = df.loc[df['punctuation_count']>0, 'comment'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in string.punctuation]))

df['punctuation_count'] = df['comment'].astype(str).apply(lambda x: len([word for word in x.split() if word in string.punctuation]))
df['punctuation_count'].value_counts().sort_index()

# checking alphabet count
most_frequent_alpha = Counter(' '.join(df['comment']))
most_frequent_alpha = pd.DataFrame(most_frequent_alpha.items(), columns=['letter', 'count'])
most_frequent_alpha['letter'].unique().tolist()[::-1]
# so many non english letters like chinese and symbols
# we have to remove them
df['comment'] = df['comment'].astype(str).apply(lambda x: ' '.join([word for word in x.split() if all(char in string.ascii_letters+' '+string.digits for char in word)]))

# again checking alphabet count
most_frequent_alpha = Counter(' '.join(df['comment']))
most_frequent_alpha = pd.DataFrame(most_frequent_alpha.items(), columns=['letter', 'count'])
most_frequent_alpha['letter'].unique().tolist()[::-1]

df['category'] = df['category'].astype(int)
df['comment'] = df['comment'].astype(str)
df.to_csv("cleaned_data.csv", index=False)
# ---------------------------------------------------------------------------------



set([word for word in df['comment'].iloc[0].split() if word in stopwords.words('hinglish')]).difference(set([word for word in df['comment'].iloc[0].split() if word in stopwords.words('english')]))

stopwords.words('hinglish')


import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
english_stopwords = nlp.Defaults.stop_words
hindi_stopwords = set(stopwords.words('hindi'))

# Keep only relevant English words for Hinglish
allowed_hinglish_english = {'like', 'come', 'still'}  # Add as needed
hinglish_stopwords = hindi_stopwords.union(english_stopwords & allowed_hinglish_english)




# distilbert-base-uncased-finetuned-sst-2-english
from transformers import pipeline
import tensorflow as tf

model3 = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
def get_sentiment(sentiment):
    if sentiment == 'POSITIVE': return 1
    elif sentiment == 'NEGATIVE': return -1
    elif sentiment == 'NEUTRAL': return 0
    else: return 0
def analyze(text):
    result = model3(text)[0]
    return get_sentiment(result["label"])
analyze(df['comment'].iloc[0])
df.iloc[0]
df['prediction'] = df['comment'].apply(lambda x: analyze(x)['sentiment'])
sample = df.sample(100)
sample['prediction'] = sample['comment'].apply(lambda x: analyze(x))
print(accuracy_score(sample['prediction'], sample['category']))
print(classification_report(sample['prediction'], sample['category']))
print(confusion_matrix(sample['prediction'], sample['category']))

sample

# -----------------------------------------------------------------------------------
# using mobilebert sentiment analysis
from transformers import pipeline
mobilebert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

result = mobilebert_pipeline("I hate this app!")
print(result) 

result = mobilebert_pipeline("I love this app!")
print(result) 

result = mobilebert_pipeline("good")
print(result) 

sample['prediction'] = sample['comment'].apply(lambda x: analyze_vader(x))

# vader sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_vader(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)
    if score['compound'] >= 0.5:
        return 1
    elif score['compound'] < 0.5 and score['compound'] > -0.5:
        return 0
    elif score['compound'] <= -0.5:
        return -1

analyze_vader("")

analyze_vader("movie bahut bekar thi")
result = model("bahut ok movie hai")
print(result)

# we will use vader sentiment analysis
import mlflow
mlflow.set_tracking_uri("http://13.127.151.52:5000/")
with mlflow.start_run():
    mlflow.log_param("model", "vader")

# using textblob sentiment analysis
from textblob import TextBlob
def analyze_textblob(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return 1
    elif blob.sentiment.polarity == 0:
        return 0
    else:
        return -1

sample = df.sample(60000)
sample.info()
%time sample['prediction'] = sample['comment'].apply(lambda x: analyze_textblob(x))
print(accuracy_score(sample['prediction'], sample['category']))
print(classification_report(sample['prediction'], sample['category']))
print(confusion_matrix(sample['prediction'], sample['category']))


%time sample['prediction2'] = sample['comment'].apply(lambda x: analyze_vader(x))
print(accuracy_score(sample['prediction2'], sample['category']))
print(classification_report(sample['prediction2'], sample['category']))
print(confusion_matrix(sample['prediction2'], sample['category']))


import fasttext
import urllib
# downloading sentiment.ftz from https://fasttext.cc/docs/en/crawl-vectors.html
backup_url = "https://www.dropbox.com/scl/fi/xyz123/sentiment.ftz?rlkey=abc123&dl=1"  # Example
urllib.request.urlretrieve(backup_url, "/home/naman/.fasttext/sentiment.ftz")


model = fasttext.load_model("/home/naman/.fasttext/sentiment.ftz")
# --------------------------------------------------------------------------------
from transformers import pipeline

# Best multilingual sentiment model
classifier = pipeline("sentiment-analysis", "nlptown/bert-base-multilingual-uncased-sentiment")

print("1. ", classifier("Ye product toh bekaar hai! 😡"))  # Negative
print("2. ",classifier("Mast kaam kiya!"))  # Positive

analyzer = pipeline("text-classification", model="l3cube-pune/hing-bert")
print("1. ", analyzer("Ye product toh bekaar hai! 😡"))  # Negative
print("2. ",analyzer("Mast kaam kiya!"))  # Positive

analyzer = pipeline("text-classification", model="l3cube-pune/hinglish-sentiment")



from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
model_name = "ganeshkharad/gk-hinglish-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelFodf.srSequenceClassification.from_pretrained(model_name)
if torch.cuda.is_available():
    model.to("cuda").half().half()
    print("Model moved to GPU and using half-precision (FP16).")
else:
    print("GPU not available, running on CPU. Performance will be slower.")
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, truncation = True, batch_size=32)

df = pd.read_csv("cleaned_data.csv")
print(sentiment_pipeline("movie was awesome"))
print(sentiment_pipeline("ye sarkar ne pakistan ko maza chakhaya kiya hai, maza aa gya"))
print(sentiment_pipeline("movie thik thi"))
print(sentiment_pipeline("movie bahut bekar thi"))
print(sentiment_pipeline("movie bahut ghatiya thi"))
print(sentiment_pipeline("this thing was not good"))
print(sentiment_pipeline("this thing was okay"))

def analyze(text):
    result = sentiment_pipeline(text)[0]
    if result['label'] == 'LABEL_2': return 1
    elif result['label'] == 'LABEL_1': return 0
    else: return -1

sample = df.sample(10000)
sample.info()
import time
t1 = time.time()
preds = sentiment_pipeline(sample['comment'].tolist())
t2 = time.time()
print(t2-t1)
%time sample['prediction'] = sample['comment'].apply(lambda x: analyze(x))

print(accuracy_score(sample['prediction'], sample['category']))
print(classification_report(sample['prediction'], sample['category'])) 

sample.sample(10)[['comment', 'category', 'prediction']]
sample['prediction2'] = sample['comment'].apply(lambda x: analyze_vader(x))
print(accuracy_score(sample['prediction2'], sample['prediction']))
print(classification_report(sample['prediction2'], sample['prediction']))

# ------------------------------------------------------------------------

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Define the model name
# Changed model_name to the one you provided: "pascalrai/hinglish-twitter-roberta-base-sentiment"
model_name2 = "pascalrai/hinglish-twitter-roberta-base-sentiment"

# Load tokenizer and model
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
model2 = AutoModelForSequenceClassification.from_pretrained(model_name2)

# Move model to GPU if available and use half-precision for faster inference
# For MX230, half-precision (FP16) is crucial for memory efficiency.
if torch.cuda.is_available():
    model.to("cuda").half()
    print("Model moved to GPU and using half-precision (FP16).")
else:
    print("GPU not available, running on CPU. Performance will be slower.")

# Initialize the sentiment analysis pipeline with explicit max_length and padding
# max_length is set to 512, which is typical for BERT/RoBERTa-based models.
# padding='max_length' ensures all sequences are padded to this length.
# truncation=True ensures longer sequences are cut.
sentiment_pipeline2 = pipeline(
    "sentiment-analysis",
    model=model2,
    tokenizer=tokenizer2, max_length=400,
    padding='max_length',
    truncation=True)

# Load your data
# Make sure 'cleaned_data.csv' is in the same directory or provide the full path
try:
    df = pd.read_csv("cleaned_data.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'cleaned_data.csv' not found. Please ensure the file is in the correct directory.")
    exit() # Exit if the file is not found

# Take a sample of 10,000 comments for analysis
sample = df.sample(100, random_state=42) # Added random_state for reproducibility
print(f"Analyzing {len(sample)} comments.")

# --- Optimized Analysis Function ---
# This function processes a list of texts in batches using the pipeline.
def analyze(texts):
    """
    Analyzes a list of texts for sentiment using the pre-initialized pipeline.
    Maps pipeline labels to numerical sentiment scores:
    LABEL_2 -> 1 (Positive)
    LABEL_1 -> 0 (Neutral)
    LABEL_0 -> -1 (Negative)
    """
    # Use torch.no_grad() for inference to reduce memory consumption and speed up calculations
    # as gradients are not needed during prediction.

    result = sentiment_pipeline(texts)

    if result[0].get('label') == 'neutral':
        return 0
    elif result[0].get('label') == 'positive':
        return 1
    else:
        return -1

# Apply the optimized batch analysis
print("Starting sentiment analysis with batch processing...")
# The pipeline handles the internal batching based on the 'batch_size' parameter.
%time sample['prediction'] = sample['comment'].apply(lambda x: analyze(x))

print("\nAnalysis complete. Displaying first few predictions:")
print(sample[['comment', 'prediction']].head())

# --------------------------------------------------------------------------------

from transformers import pipeline

# 1. Load the sentiment analysis pipeline with the specific model
# The pipeline abstracts away much of the complexity.
# The 'text-classification' task is used for sentiment analysis.
try:
    classifier = pipeline(
        "text-classification",
        model="AmeenKhan/Sentiment-Analysis-Code-Mixed-Hinglish-Text", truncation = True, batch_size=32
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an active internet connection and the model name is correct.")
    print("You might also try pip install transformers[torch] or transformers[tensorflow] based on your setup.")
    exit()

# 4. Print the results
for i, text in enumerate(hinglish_texts):
    label = results[i]['label']
    score = results[i]['score']
    print(f"\nText: \"{text}\"")
    print(f"Sentiment: {label} (Score: {score:.4f})")

print("\n--- Analysis Complete ---")

# You can also analyze a single text:
single_text = "Mausam to mast hai bhai!"
single_result = classifier(single_text)
print(f"\nSingle Text: \"{single_text}\"")
print(f"Sentiment: {single_result[0]['label']} (Score: {single_result[0]['score']:.4f})")


sample = df.sample(10000)
import time
t1 = time.time()
preds = classifier(sample['comment'].tolist())
t2 = time.time()



import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "ganeshkharad/gk-hinglish-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Move model to CPU first, as quantization is often done on CPU
model.eval() # Set model to evaluation mode

# Perform Dynamic Quantization (easiest for inference)
# This quantizes the Linear and Conv layers
quantized_model = model.half()

print("Original Model Size (MB):", sum(p.numel() for p in model.parameters()) * 4 / (1024**2))
print("Quantized Model Size (MB):", sum(p.numel() for p in quantized_model.parameters()) * 2 / (1024**2)) # Approx 1 byte per param for INT8

# Now you can try to move the quantized model to GPU if it fits
# This might still be too large for 2GB, but it's a significant reduction
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantized_model.to(device)
    print(f"Quantized model moved to {device}")

    # Test with an example
    text = "Yeh bahut achha hai."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = quantized_model(**inputs)
    print(f"Quantized model output: {outputs.logits}")

except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("Quantized model still too large for GPU, staying on CPU.")
        # In this case, you might need to use imCPU for inference, or
        # explore model pruning/distillation (next steps).
    else:
        raise e

torch.cuda.is_available()  # Check if CUDA is available
# For saving the quantized model:
# torch.save(quantized_model.state_dict(), "quantized_gk_hinglish_sentiment.pth")
# For loading, you'd need to re-quantize the model structure, then load state_dict
# (more complex, see PyTorch quantization docs for full workflow)








# ------------ working ------------

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
model_name = "ganeshkharad/gk-hinglish-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Convert Model to Half-Precision (FP16)
    half_precision_model = model.half()

    # Set Device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        half_precision_model.to(device)
        half_precision_model.eval() # Set model to evaluation mode
        print(f"Model '{model_name}' loaded and converted to FP16. Running on: {device}")
    else:
        half_precision_model = half_precision_model.half()
        half_precision_model.to(device)
        print(f"Model '{model_name}' loaded and converted to FP8. Running on: {device}")

except Exception as e:
    print(f"Error loading or converting model: {e}")
    print("Please ensure you have an active internet connection, 'transformers' and 'torch' installed.")
    exit()

# --- 2. Create the Sentiment Analysis Pipeline ---
# You pass your converted model and tokenizer directly to the pipeline.
# The pipeline will automatically handle input/output processing.
# We explicitly set device=device to ensure it runs on GPU if available.
sentiment_pipeline = pipeline(
    "text-classification",
    model=half_precision_model,
    tokenizer=tokenizer, truncation=True,
    device=0 if device.type == 'cuda' else -1
)

sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer, truncation=True, max_length= 400,
    device=0 if torch.cuda.is_available() else -1)


import pandas as pd
df = pd.read_csv("cleaned_data.csv")
sample = df.sample(2000)

import time
t1 = time.time()
preds = sentiment_pipeline2(sample['comment'].tolist())
print(f"Time taken for sentiment analysis: {(time.time() - t1)/60:.2f} minutes")



sentiment_pipeline("As a maths teacher, i stand with your vision and intellectuality. Such a noble gesture to not criticize the others shown by you is highly appreciated. More power to you")
sentiment_pipeline2("As a maths teacher, i stand with your vision and intellectuality. Such a noble gesture to not criticize the others shown by you is highly appreciated. More power to you")[0]['label']
analyze_vader("As a maths teacher, i stand with your vision and intellectuality. Such a noble gesture to not criticize the others shown by you is highly appreciated. More power to you")
analyze_textblob("As a maths teacher, i stand with your vision and intellectuality. Such a noble gesture to not criticize the others shown by you is highly appreciated. More power to you")
model3("As a maths teacher, i stand with your vision and intellectuality. Such a noble gesture to not criticize the others shown by you is highly appreciated. More power to you")
comment1 = "As a maths teacher, i stand with your vision and intellectuality. Such a noble gesture to not criticize the others shown by you is highly appreciated. More power to you"
sentiment_pipeline(" ".join([x for x in comment1.split() if x not in stop_words+[]]))

from nltk.corpus import stopwords
stop_words2 = stopwords.words('hinglish')
stop_words = stopwords.words('english')

sentiment_pipeline2('wow')



import pandas as pd
df2 = pd.read_csv('../youtube_comments_with_sentiment.csv')
df3 = df2[['text_original', 'cleaned_comment', 'sentiment_label', 'sentiment_score']]
show(df3)
df2.head()

 
from pandasgui import show
show(pd.DataFrame(negative_words, columns=['negative_words']))

stop_words2 = pd.DataFrame(stop_words2, columns=['a'])
# query to get result ending with "n't" using query
negative_words = [i for i in stop_words if i.endswith("n't")] + [i for i in stop_words if i.endswith("n") and len(i) > 1 and i[-2].lower() not in "aeiou"] + ['weren', 'no', 'not', 'nor']

comment2 = "jai shree ram"
ignore_words = [i for i in stop_words if i not in negative_words]

analyze_textblob(' '.join([i for i in comment2.split() if i not in ignore_words]))
analyze_textblob('thank you')
analyze_textblob('okay')

df3['blob'] = df3['cleaned_comment'].apply(lambda x: sentiment_pipeline2(x)[0]['label'])
show(df3)
sentiment_pipeline2('thanks')

def analyze_textblob2(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return "LABEL_2"
    elif blob.sentiment.polarity == 0:
        return "LABEL_1"
    else:
        return "LABEL_0"




import re

def contains_devanagari(text):
    """
    Checks if a string contains any Devanagari (Hindi) characters.
    Devanagari Unicode range: U+0900 to U+097F
    """
    # Regex to match any character in the Devanagari Unicode block
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    return bool(devanagari_pattern.search(text))

# Example Usage:
comment1 = "आज मौसम बहुत अच्छा है. This is great!" # Hinglish with Devanagari
comment2 = "Kya haal hai? This is good." # Transliterated Hinglish
comment3 = "This is just an English comment." # Pure English

print(f"'{comment1}' contains Devanagari: {contains_devanagari(comment1)}")
print(f"'{comment2}' contains Devanagari: {contains_devanagari(comment2)}")
print(f"'{comment3}' contains Devanagari: {contains_devanagari(comment3)}")

# Output:
# 'आज मौसम बहुत अच्छा है. This is great!' contains Devanagari: True
# 'Kya haal hai? This is good.' contains Devanagari: False
# 'This is just an English comment.' contains Devanagari: False