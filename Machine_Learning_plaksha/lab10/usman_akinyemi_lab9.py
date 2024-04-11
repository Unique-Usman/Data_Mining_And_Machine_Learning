"""
Lab 9 Machine Learning Assignment
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#=========================================================================================
#nltk.download('punkt')
#nltk.download('stopwords')
#=========================================================================================
np.random.seed(0)
#=========================================================================================
data = pd.read_excel('factoryReports.xlsx')
stop_words = set(stopwords.words('english'))
data['Description'] = data['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stop_words]))
data['Category'] = data['Category'].str.lower()
#=========================================================================================
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(data['Description'])
y = data['Category']
#=========================================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#=========================================================================================
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
#=========================================================================================
y_pred = rf_classifier.predict(X_test)
#=========================================================================================
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
