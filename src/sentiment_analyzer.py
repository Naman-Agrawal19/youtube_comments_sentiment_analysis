# src/sentiment_analyzer.py

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.amp import autocast
import time

def initialize_sentiment_pipeline(model_name, max_length=256, pipeline_batch_size=256):
    """
    Loads the model, converts to FP16, and initializes the Hugging Face pipeline.
    This function should be called only once.

    Args:
        model_name (str): Hugging Face model identifier.
        max_length (int): Max token length for model input.
        pipeline_batch_size (int): Batch size for pipeline inference.

    Returns:
        transformers.Pipeline: The initialized sentiment analysis pipeline.
    Raises:
        RuntimeError: If failed to initialize the pipeline.
    """
    print(f"Initializing sentiment model '{model_name}'...")
    try:
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the base model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Convert the model to half-precision (FP16) for memory and speed
        # half_precision_model = model.half()
        # half_precision_model.to(device)
        # half_precision_model.eval() # Set to evaluation mode

        # Initialize the Hugging Face pipeline
        sentiment_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=max_length,
            device=-1 # 0 for GPU 0, -1 for CPU
        )
        print("Sentiment pipeline initialized successfully.")
        return sentiment_pipeline
    except Exception as e:
        raise RuntimeError(f"Failed to initialize sentiment pipeline: {e}")

def analyze_sentiment(sentiment_pipeline, texts, pipeline_batch_size):
    """
    Performs sentiment analysis on a list of texts using the initialized pipeline.

    Args:
        sentiment_pipeline (transformers.Pipeline): The initialized sentiment analysis pipeline.
        texts (list): A list of strings (comments) to analyze.
        pipeline_batch_size (int): Batch size to use for pipeline inference.

    Returns:
        list: A list of dictionaries, each containing 'label' and 'score' for a text.
    """
    if not sentiment_pipeline:
        raise RuntimeError("Sentiment pipeline not initialized.")

    print(f"\nStarting sentiment analysis on {len(texts)} comments...")
    start_time = time.time()

    # Determine device from the pipeline's internal model
    # device = sentiment_pipeline.device.type

    # # Use autocast for mixed precision if on CUDA. The pipeline handles batching.
    # if device == 'cuda':
    #     with autocast():
    #         results = sentiment_pipeline(texts, batch_size=pipeline_batch_size)
    # else:
    results = sentiment_pipeline(texts, batch_size=pipeline_batch_size)

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Sentiment analysis completed in {processing_time:.2f} seconds.")
    print(f"Average time per comment: {processing_time / len(texts):.4f} seconds.")

    return results