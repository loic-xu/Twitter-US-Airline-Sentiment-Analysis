# Twitter US Airline Sentiment Analyzer

This project aims to analyze sentiments expressed in tweets about US airlines using Machine Learning techniques. The goal is to predict whether a tweet is **positive**, **neutral**, or **negative** by testing various models and Natural Language Processing (NLP) approaches.

## 📝 Context

Passengers' sentiments toward airlines can significantly influence their loyalty. In a highly competitive industry, understanding and improving customer satisfaction is crucial. This project explores sentiment analysis on Twitter to help airlines better gauge customer satisfaction.

## 🔍 Objectives

- Test different Machine Learning models for sentiment classification (Logistic Regression, Naive Bayes, Random Forest, RNN, BERT).
- Compare model performances in terms of accuracy and computational cost.
- Identify the best feature combinations to improve prediction accuracy.

## 📊 Models Tested

1. **Multinomial Logistic Regression**: A simple and effective model for multiclass classification.
2. **Naive Bayes**: Based on Bayes' theorem to estimate class probabilities.
3. **Random Forest**: An ensemble of Decision Trees to improve prediction robustness.
4. **VADER**: A library specifically designed for sentiment analysis on social media.
5. **BERT and its derivatives (RoBERTa, DistilBERT)**: State-of-the-art models for Natural Language Processing.


## 📈 Results

- **Best Model**: **Fine-tuned RoBERTa** with an accuracy of **84.63%** on the test set.
- **Best Cost/Performance Trade-off**: **Logistic Regression** with TF-IDF and n-grams, achieving **81.94%** accuracy.
- Random Forest and LSTM models did not perform well due to overfitting or poor generalization.

## 🛠️ Methodology

1. **Data Preprocessing**:
   - Removal of duplicates and irrelevant columns.
   - Conversion of emojis to text.
   - Tokenization, stopword removal, lemmatization.
   - Transformation of sentiments into numerical values.

2. **Feature Engineering**:
   - TF-IDF, Term Frequency, tweet length, n-grams, VADER score.

3. **Hyperparameter Tuning**:
   - Used GridSearchCV to optimize Logistic Regression and Random Forest models.
   - Cross-validation for BERT due to its high computational cost.

## 📚 References

- Cliche, M. (2017). BB_twtr at SemEval-2017 Task 4: Twitter Sentiment Analysis with CNNs and LSTMs.
- Chiorrini et al. (2021). Emotion and sentiment analysis of tweets using BERT.
- Alima et al. (2022). Sentiment Analysis using Logistic Regression.
