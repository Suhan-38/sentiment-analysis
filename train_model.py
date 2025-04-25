import nltk
import pandas as pd
import numpy as np
import re
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Download necessary NLTK resources
print("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('movie_reviews', quiet=True)

# Set of stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (but keep the text without the # symbol)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s,]', '', text)
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize text and remove stopwords
    words = nltk.word_tokenize(text)
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

# Function to create a dataset from NLTK movie reviews
def create_movie_reviews_dataset(max_reviews=1000):
    from nltk.corpus import movie_reviews
    
    print(f"Creating dataset from NLTK movie reviews (max {max_reviews} reviews per class)...")
    
    # Get positive and negative review IDs
    positive_fileids = movie_reviews.fileids('pos')[:max_reviews]
    negative_fileids = movie_reviews.fileids('neg')[:max_reviews]
    
    # Get review texts
    positive_texts = [movie_reviews.raw(fileid) for fileid in positive_fileids]
    negative_texts = [movie_reviews.raw(fileid) for fileid in negative_fileids]
    
    # Create dataset
    texts = positive_texts + negative_texts
    labels = ['positive'] * len(positive_texts) + ['negative'] * len(negative_texts)
    
    return pd.DataFrame({'text': texts, 'sentiment': labels})

# Function to create a custom dataset with additional examples
def create_custom_dataset():
    print("Creating custom dataset with additional examples...")
    
    # Custom examples with clear sentiment signals
    custom_data = {
        'text': [
            # Positive examples
            "I absolutely love this product! It's awesome",
            "Great value for money. Delivered on time.",
            "Fantastic product, highly recommend it!",
            "I'm so happy with my purchase, it works perfectly!",
            "This makes me happy every time I use it. Best decision ever.",
            "Excellent service and quality. Very happy customer here!",
            "The team was friendly and helpful. Happy to recommend them.",
            "Delighted with how quickly it arrived. Great product!",
            "This product brings joy to my daily routine. Love it!",
            "Very satisfied with the performance. Will buy again!",
            "The customer service was outstanding and prompt.",
            "This exceeded all my expectations. Truly amazing!",
            "A perfect solution to my problem. Thank you!",
            "Incredible quality for the price. Great value!",
            "I'm impressed with how well this works. Brilliant!",
            
            # Negative examples
            "Worst service ever! Never buying again.",
            "Not satisfied... Took ages to deliver.",
            "Terrible experience. Do not recommend.",
            "Disappointed with the quality. Complete waste of money.",
            "This product made me angry. It broke after one use.",
            "Frustrated with customer service. No help at all.",
            "Sad to say this was the worst purchase I've made.",
            "Regret buying this. It doesn't work as advertised.",
            "Unhappy with the entire experience. Avoid this company.",
            "This is awful. Don't waste your money like I did.",
            "Poor quality and overpriced. Very disappointed.",
            "The product arrived damaged and customer service was unhelpful.",
            "Completely useless for its intended purpose. Avoid!",
            "Terrible design and even worse functionality.",
            "This is the worst product I've ever bought. Absolute garbage."
        ],
        'sentiment': [
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'positive', 'positive', 'positive', 'positive', 'positive',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative',
            'negative', 'negative', 'negative', 'negative', 'negative'
        ]
    }
    
    return pd.DataFrame(custom_data)

# Main function to train and save the model
def train_and_save_model():
    # Create datasets
    movie_df = create_movie_reviews_dataset(max_reviews=500)
    custom_df = create_custom_dataset()
    
    # Combine datasets
    df = pd.concat([movie_df, custom_df], ignore_index=True)
    print(f"Combined dataset size: {len(df)} examples")
    
    # Preprocess text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Create and fit vectorizer
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    with open('models/sentiment_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save common positive and negative words for direct classification
    positive_words = ['happy', 'good', 'great', 'excellent', 'amazing', 'love', 
                      'joy', 'delighted', 'awesome', 'fantastic', 'wonderful', 
                      'brilliant', 'perfect', 'outstanding', 'superb', 'best',
                      'pleased', 'satisfied', 'impressive', 'exceptional']
    
    negative_words = ['sad', 'bad', 'terrible', 'awful', 'horrible', 'hate', 
                      'angry', 'disappointed', 'poor', 'worst', 'useless', 
                      'frustrating', 'annoying', 'pathetic', 'disgusting', 
                      'dreadful', 'mediocre', 'inferior', 'appalling', 'atrocious']
    
    word_sentiments = {
        'positive_words': positive_words,
        'negative_words': negative_words
    }
    
    with open('models/word_sentiments.pkl', 'wb') as f:
        pickle.dump(word_sentiments, f)
    
    print("Training and saving complete!")
    return model, vectorizer, word_sentiments

if __name__ == "__main__":
    train_and_save_model()
