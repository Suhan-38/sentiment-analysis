from flask import Flask, render_template, request, jsonify, url_for, redirect
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import re
import string

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Set of stopwords
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Create directories for static files
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Sample data for initial model training
sample_data = {
    'text': [
        "I absolutely love this product! 😍 #awesome",
        "Worst service ever! @company 😡 Never buying again.",
        "Great value for money. Delivered on time. 👏",
        "Not satisfied... Took ages to deliver. 👎 #badservice",
        "Fantastic product, highly recommend it! 🔥",
        "Terrible experience. Do not recommend. 🤮"
    ],
    'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']
}

# Preprocessing function
def preprocess_social_media_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (but keep the text without the # symbol)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove emojis
    text = re.sub(r'[^\w\s,]', '', text)
    
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Tokenize text and remove stopwords
    words = nltk.word_tokenize(text)
    text = ' '.join([word for word in words if word not in stop_words])
    
    return text

# Function to train model on sample data
def train_model():
    df = pd.DataFrame(sample_data)
    df['text'] = df['text'].apply(preprocess_social_media_text)
    
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    # Train Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)
    
    return model, vectorizer, df

# Function to analyze new text
def analyze_text(text, model, vectorizer):
    # Preprocess the text
    processed_text = preprocess_social_media_text(text)
    
    # Vectorize the text
    text_vec = vectorizer.transform([processed_text])
    
    # Predict sentiment
    sentiment = model.predict(text_vec)[0]
    
    # Get probability scores
    proba = model.predict_proba(text_vec)[0]
    confidence = proba[list(model.classes_).index(sentiment)]
    
    return sentiment, confidence, processed_text

# Function to generate confusion matrix
def generate_confusion_matrix(model, vectorizer, df):
    X = vectorizer.transform(df['text'])
    y_true = df['sentiment']
    y_pred = model.predict(X)
    
    conf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Save to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    encoded_img = base64.b64encode(img_data.getvalue()).decode('utf-8')
    return encoded_img

# Function to generate word cloud
def generate_wordcloud(text, sentiment):
    if sentiment == 'positive':
        colormap = 'viridis'
        bg_color = 'white'
    else:
        colormap = 'Purples'
        bg_color = 'black'
    
    wordcloud = WordCloud(width=800, height=400,
                         background_color=bg_color,
                         colormap=colormap,
                         max_font_size=150,
                         random_state=42).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    
    # Save to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    encoded_img = base64.b64encode(img_data.getvalue()).decode('utf-8')
    return encoded_img

# Function to generate top words visualization
def generate_top_words(model, vectorizer, sentiment):
    feature_names = vectorizer.get_feature_names_out()
    class_idx = list(model.classes_).index(sentiment)
    
    # Get top 20 words
    top_words_idx = np.argsort(model.feature_log_prob_[class_idx])[-20:]
    top_words = [feature_names[i] for i in top_words_idx]
    log_probs = model.feature_log_prob_[class_idx, top_words_idx]
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_words, log_probs)
    plt.title(f'Top 20 Words for {sentiment.capitalize()} Sentiment')
    plt.xlabel('Log Probability')
    plt.tight_layout()
    
    # Save to BytesIO object
    img_data = BytesIO()
    plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
    img_data.seek(0)
    plt.close()
    
    # Convert to base64 for embedding in HTML
    encoded_img = base64.b64encode(img_data.getvalue()).decode('utf-8')
    return encoded_img

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Get user input
        user_text = request.form.get('text', '')
        
        if not user_text:
            return jsonify({'error': 'Please enter some text to analyze'})
        
        # Train model (in a real app, you'd do this once and cache it)
        model, vectorizer, df = train_model()
        
        # Analyze the text
        sentiment, confidence, processed_text = analyze_text(user_text, model, vectorizer)
        
        # Add the new text to our dataset for visualization purposes
        new_df = df.copy()
        new_df = pd.concat([new_df, pd.DataFrame({
            'text': [processed_text],
            'sentiment': [sentiment]
        })], ignore_index=True)
        
        # Generate visualizations
        conf_matrix_img = generate_confusion_matrix(model, vectorizer, new_df)
        
        # Get all texts of the predicted sentiment
        sentiment_texts = ' '.join(new_df[new_df['sentiment'] == sentiment]['text'])
        wordcloud_img = generate_wordcloud(sentiment_texts, sentiment)
        
        # Generate top words visualization
        top_words_img = generate_top_words(model, vectorizer, sentiment)
        
        return render_template('results.html', 
                              text=user_text,
                              sentiment=sentiment,
                              confidence=f"{confidence:.2f}",
                              conf_matrix_img=conf_matrix_img,
                              wordcloud_img=wordcloud_img,
                              top_words_img=top_words_img)

if __name__ == '__main__':
    app.run(debug=True)
