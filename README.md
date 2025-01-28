# Restaurant-Reviews-Sentiment-Analysis

## Overview
This project applies sentiment analysis techniques to a dataset of restaurant reviews. The goal is to classify reviews as either **positive** or **negative** using various machine learning algorithms like Logistic Regression, Naive Bayes, Random Forest, and others. The results provide insights into customer satisfaction and help restaurant managers improve their services and reputation.

---

## Technologies Used
- **Python**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **WordCloud**
- **XGBoost**
- **Jupyter Notebook**

---

## Getting Started
### Prerequisites
- **Python 3.x**
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
### Dataset: 
The dataset **Restaurant_reviews.tsv** consists of restaurant reviews and their corresponding sentiment labels (Liked column). You can upload your dataset or use a sample similar to the one used in this project.

---

## Usage
### 1. Clone the Repository:
```bash
git clone https://github.com/ptrishita/Restaurant-Reviews-Sentiment-Analysis.git
```
### 2. Run the notebook:
```bash
jupyter notebook restaurant_sentiment_analysis.ipynb
```
### 3. Modify the dataset:
- Update the dataset path in the code if you are using your own dataset.
- Ensure the dataset has two columns: Review (text of the review) and Liked (sentiment label).
### 4. Run the analysis:
The project will automatically
- Clean and preprocess the data.
- Vectorize the text using TF-IDF.
- Train multiple machine learning models.
- Evaluate model performance.
- Visualize sentiment distributions using pie charts and word clouds.

---

## Models Implemented
This project uses the following machine learning models for sentiment classification:
- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)

Each model is evaluated on accuracy, precision, recall, and F1-score to determine the best performing model for sentiment classification.

---

## Future Scope
- **Expanding Sentiment Categories:** Improve the model by adding more granular sentiment categories such as very positive, neutral, etc.
- **Real-time Sentiment Analysis:** Implement real-time sentiment analysis by integrating with social media platforms or review sites.
- **Multilingual Support:** Expand the model to support multiple languages for analyzing reviews from different regions.
- **Aspect-Based Sentiment Analysis:** Perform analysis based on specific aspects of the review (e.g., service, food, ambiance).
- **Integration with Restaurant Systems:** Integrate sentiment analysis with restaurant management systems to automate feedback processing and action.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
