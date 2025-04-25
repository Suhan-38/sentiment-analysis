# Sentiment Analysis Web Application

This project performs sentiment analysis on social media text and generates various visualizations to help understand the model and data. It includes both a script version for viewing visualizations in VSCode and a web application for interactive analysis.

## Project Overview

The project includes:
- A sentiment analysis model using Naive Bayes
- Text preprocessing for social media content
- Visualization of model performance and key features
- Word clouds for positive and negative sentiments
- A web application for interactive sentiment analysis

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
   pip install -r requirements.txt
   ```

3. **Download NLTK resources** (this will happen automatically when you run the script, but you can also do it manually):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```

### Running the Project

#### Script Version

To generate all visualizations using the script version:

```bash
python script.py
```

This will:
1. Process the sample social media data
2. Train a Naive Bayes sentiment analysis model
3. Generate all visualizations in the `visualizations` folder
4. Display the model's accuracy and classification report in the console

#### Web Application

To run the web application:

```bash
# First time only: Train the model (this will happen automatically if needed)
python train_model.py

# Run the web application
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000/
```

The web application allows you to:
1. Enter any text for sentiment analysis
2. View the predicted sentiment with confidence score
3. See interactive visualizations of the analysis results
4. Analyze multiple texts in a user-friendly interface

**Note**: The model is pre-trained and saved, so you don't need to retrain it each time you run the application. The training data includes NLTK movie reviews and custom examples for better accuracy.

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

### Script Version
You can modify the `script.py` file to:
- Add your own dataset by changing the `data` dictionary
- Adjust the preprocessing steps in the `preprocess_social_media_text` function
- Change visualization parameters like colors, sizes, and styles
- Save visualizations in different formats or locations

### Web Application
You can customize the web application by:
- Modifying the HTML templates in the `templates` folder to change the UI
- Adjusting the Flask routes in `app.py` to add new features
- Changing the visualization styles in the Python code
- Adding new types of analysis or visualizations

## Troubleshooting

Common issues:

1. **NLTK Resource Errors**: If you encounter errors about missing NLTK resources, make sure you've downloaded them using the commands in the Installation section.

2. **Visualization Not Showing**: If visualizations don't appear when running the script, check that matplotlib is properly configured for your environment.

3. **Package Not Found**: If you get "module not found" errors, ensure you've installed all required packages listed in the Installation section.

4. **Web Application Issues**: If the Flask application doesn't start, check that you have Flask installed and that port 5000 is not in use by another application.

## License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).
