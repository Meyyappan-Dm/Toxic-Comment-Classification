# 🧠 Multi-Label Toxic Comment Classification

This project builds a **multi-label text classification model** that identifies different types of toxicity in user comments. Using **TF-IDF** vectorization and machine learning algorithms like **Logistic Regression** and **Multinomial Naive Bayes**, the system predicts multiple labels such as *toxic*, *severe toxic*, *obscene*, *threat*, *insult*, and *identity hate* for each comment.

---

## 🚀 Features

- Preprocesses and cleans raw text (lowercasing, removing stopwords, stemming)
- Handles **multi-label classification** where comments may have multiple toxicity types
- Implements two powerful models:
  - **Multinomial Naive Bayes**
  - **Logistic Regression**
- Automatically evaluates model performance using:
  - ROC-AUC score  
  - Accuracy metrics  
  - Classification Report
- Visualizations for:
  - Label distribution  
  - Number of labels per comment  
  - ROC Curves for each label

---

## ⚙️ How It Works

1. **Load the dataset**  
   Reads the training data containing text and label columns.

2. **Preprocess text**  
   - Cleans punctuation and contractions  
   - Removes stopwords  
   - Applies stemming  

3. **Split dataset**  
   Divides data into training (80%) and test (20%) sets.

4. **Build classification pipelines**
   - Uses `TfidfVectorizer` for text to numerical feature transformation
   - Wraps classifiers in `OneVsRestClassifier` for multi-label learning

5. **Train and evaluate models**
   - Compares Naive Bayes and Logistic Regression  
   - Outputs ROC-AUC and accuracy  
   - Displays per-label metrics  

6. **Plot results**
   - Label distribution with Seaborn barplots  
   - ROC curves per class using Matplotlib

---

## 📊 Model Evaluation

Example output includes:

- **ROC-AUC Score:** 0.93  
- **Accuracy:** 0.87  


