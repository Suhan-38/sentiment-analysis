import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import stopwords
import re
import string
from wordcloud import WordCloud
import os

# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)  # This is needed for tokenization

# Set of stopwords
stop_words = set(stopwords.words('english'))

# Sample Social Media Dataset
data = {
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

# Convert to DataFrame
df = pd.DataFrame(data)

# Enhanced Preprocessing Function
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

# Apply preprocessing
df['text'] = df['text'].apply(preprocess_social_media_text)

# Split features and labels
X = df['text']
y = df['sentiment']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))  # Include bigrams for richer features
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

print('Classification Report:')
print(classification_report(y_test, y_pred))

# Display Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualize Top Words
def plot_top_words(vectorizer, model, n=20):
    feature_names = vectorizer.get_feature_names_out()
    class_labels = model.classes_

    for i, label in enumerate(class_labels):
        top_words = np.argsort(model.feature_log_prob_[i])[-n:]
        plt.figure(figsize=(10, 8))
        plt.barh([feature_names[j] for j in top_words], model.feature_log_prob_[i, top_words])
        plt.title(f'Top {n} Words for Class: {label}')
        plt.xlabel('Log Probability')
        plt.tight_layout()
        plt.savefig(f'visualizations/top_words_{label}.png', dpi=300, bbox_inches='tight')
        plt.show()

plot_top_words(vectorizer, model)

# Generate WordClouds
positive_text = ' '.join(df[df['sentiment'] == 'positive']['text'])
negative_text = ' '.join(df[df['sentiment'] == 'negative']['text'])

# Positive WordCloud
wordcloud_positive = WordCloud(width=800, height=400,
                              background_color='white',
                              colormap='viridis',
                              max_font_size=150,
                              random_state=42).generate(positive_text)

plt.figure(figsize=(10, 5))
plt.title('Positive Sentiment WordCloud')
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('visualizations/positive_wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# Negative WordCloud
wordcloud_negative = WordCloud(width=800, height=400,
                              background_color='black',
                              colormap='Purples',
                              max_font_size=150,
                              random_state=42).generate(negative_text)

plt.figure(figsize=(10, 5))
plt.title('Negative Sentiment WordCloud')
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig('visualizations/negative_wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# Print information about saved visualizations
print("\nAll visualizations have been saved to the 'visualizations' folder:")
print("- confusion_matrix.png")
print("- top_words_positive.png")
print("- top_words_negative.png")
print("- positive_wordcloud.png")
print("- negative_wordcloud.png")