# Sentiment Analysis Visualizations

This project performs sentiment analysis on social media text and generates various visualizations to help understand the model and data. The visualizations are designed to be viewed directly in VSCode without requiring a web browser.

## Project Overview

The project includes:
- A sentiment analysis model using Naive Bayes
- Text preprocessing for social media content
- Visualization of model performance and key features
- Word clouds for positive and negative sentiments

## Getting Started

### Prerequisites

To run this project, you need:

- Python 3.6 or higher
- Required Python packages (see Installation)
- Visual Studio Code (for optimal visualization viewing)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Suhan-38/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install required packages**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn nltk wordcloud
   ```

3. **Download NLTK resources** (this will happen automatically when you run the script, but you can also do it manually):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

### Running the Project

To generate all visualizations:

```bash
python script.py
```

This will:
1. Process the sample social media data
2. Train a Naive Bayes sentiment analysis model
3. Generate all visualizations in the `visualizations` folder
4. Display the model's accuracy and classification report in the console

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
   - Install the "Image Preview" extension for enhanced image viewing capabilities:
     - Open VSCode
     - Go to Extensions (Ctrl+Shift+X)
     - Search for "Image Preview"
     - Click Install
   - This extension allows you to hover over image paths in your code to see a preview.

## Customizing the Project

You can modify the `script.py` file to:
- Add your own dataset by changing the `data` dictionary
- Adjust the preprocessing steps in the `preprocess_social_media_text` function
- Change visualization parameters like colors, sizes, and styles
- Save visualizations in different formats or locations

## Troubleshooting

Common issues:

1. **NLTK Resource Errors**: If you encounter errors about missing NLTK resources, make sure you've downloaded them using the commands in the Installation section.

2. **Visualization Not Showing**: If visualizations don't appear when running the script, check that matplotlib is properly configured for your environment.

3. **Package Not Found**: If you get "module not found" errors, ensure you've installed all required packages listed in the Installation section.

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
