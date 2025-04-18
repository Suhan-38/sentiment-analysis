# Sentiment Analysis Visualizations

This project performs sentiment analysis on social media text and generates various visualizations to help understand the model and data.

## Visualizations

All visualizations are saved in the `visualizations` folder:

1. **Confusion Matrix** (`confusion_matrix.png`): Shows the model's prediction accuracy.
2. **Word Clouds**:
   - `positive_wordcloud.png`: Visual representation of words in positive sentiment texts.
   - `negative_wordcloud.png`: Visual representation of words in negative sentiment texts.
3. **Top Words**:
   - `top_words_positive.png`: Bar chart showing the top 20 words for positive sentiment.
   - `top_words_negative.png`: Bar chart showing the top 20 words for negative sentiment.

## How to View Visualizations in VSCode

1. **Using VSCode's Built-in Image Preview**:
   - Simply click on any `.png` file in the Explorer panel to open it in VSCode's image viewer.
   - You can zoom in/out using the controls in the top-right corner of the image viewer.

2. **Using VSCode Extensions**:
   - Install the "Image Preview" extension for enhanced image viewing capabilities.
   - This extension allows you to hover over image paths in your code to see a preview.

## Running the Script

To regenerate all visualizations, run:

```bash
python script.py
```

This will create or update all visualization files in the `visualizations` folder.
