Binary Classification with Logistic Regression, Gaussian Naive Bayes, and Bernoulli Naive Bayes
With Jupyter Notebook

This Jupyter Notebook demonstrates how to perform binary classification using Logistic Regression, Gaussian Naive Bayes, and Bernoulli Naive Bayes. The dataset is generated using make_classification from scikit-learn, and the models are evaluated using accuracy, confusion matrices, and ROC curves.

Table of Contents
Prerequisites

Getting Started

Running the Code

Code Explanation

Results

License

Prerequisites
Before running the code, ensure you have the following installed:

Python 3.x

Required Python libraries:

bash
Copy
pip install numpy pandas scikit-learn matplotlib seaborn scikit-plot
Jupyter Notebook (to run the .ipynb file).

Getting Started
Launch Jupyter Notebook
Start Jupyter Notebook:

bash
Copy
jupyter notebook
Open the .ipynb file from the Jupyter Notebook interface.

Running the Code
Open the .ipynb file in Jupyter Notebook.

Run each cell sequentially to execute the code.

Code Explanation
1. Import Libraries
python
Copy
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, plot_roc_curve
Libraries used for data generation, visualization, modeling, and evaluation.

2. Generate Synthetic Dataset
python
Copy
x, y = make_classification(n_samples=400, n_classes=2, n_features=3, n_informative=2, n_redundant=0, n_clusters_per_class=2)
print(x[:5])
print(y[:5])
Generate a synthetic dataset with 400 samples, 3 features, and 2 classes.

3. Data Visualization
python
Copy
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
sns.histplot(x[:, 0], kde=True, ax=axes[0])
sns.histplot(x[:, 1], kde=True, ax=axes[1])
sns.histplot(x[:, 2], kde=True, ax=axes[2])
plt.show()
Visualize the distribution of each feature using histograms.

4. Train-Test Split
python
Copy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
Split the dataset into training and testing sets.

5. Train Models
python
Copy
gb = GaussianNB()
bb = BernoulliNB()
lg = LogisticRegression()

gb.fit(x_train, y_train)
bb.fit(x_train, y_train)
lg.fit(x_train, y_train)
Train Gaussian Naive Bayes, Bernoulli Naive Bayes, and Logistic Regression models.

6. Evaluate Models
python
Copy
print("GaussianNB Train Accuracy:", gb.score(x_train, y_train))
print("BernoulliNB Train Accuracy:", bb.score(x_train, y_train))
print("LogisticRegression Train Accuracy:", lg.score(x_train, y_train))

y_pred1 = gb.predict(x_test)
y_pred2 = bb.predict(x_test)
y_pred3 = lg.predict(x_test)

print("GaussianNB Test Accuracy:", accuracy_score(y_test, y_pred1))
print("BernoulliNB Test Accuracy:", accuracy_score(y_test, y_pred2))
print("LogisticRegression Test Accuracy:", accuracy_score(y_test, y_pred3))
Evaluate the models using accuracy scores.

7. Classification Reports
python
Copy
print("GaussianNB Classification Report:\n", classification_report(y_test, y_pred1))
print("BernoulliNB Classification Report:\n", classification_report(y_test, y_pred2))
print("LogisticRegression Classification Report:\n", classification_report(y_test, y_pred3))
Generate classification reports for each model.

8. Confusion Matrices
python
Copy
figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
sns.heatmap(confusion_matrix(y_test, y_pred1), annot=True, ax=axes[0])
sns.heatmap(confusion_matrix(y_test, y_pred2), annot=True, ax=axes[1])
sns.heatmap(confusion_matrix(y_test, y_pred3), annot=True, ax=axes[2])
plt.show()
Visualize confusion matrices for each model.

9. ROC Curve Comparison
python
Copy
fig, ax = plt.subplots(figsize=(8, 6))
plot_roc_curve(gb, x_test, y_test, ax=ax, name='GaussianNB')
plot_roc_curve(bb, x_test, y_test, ax=ax, name='BernoulliNB')
plot_roc_curve(lg, x_test, y_test, ax=ax, name='LogisticRegression')

ax.set_title("ROC Curve Comparison")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()
plt.show()
Plot ROC curves for each model to compare their performance.

Results
Accuracy: Test accuracy for GaussianNB, BernoulliNB, and LogisticRegression.

Classification Reports: Precision, recall, and F1-score for each model.

Confusion Matrices: Visual representation of true vs predicted values.

ROC Curves: Comparison of model performance using ROC curves.

License
This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

Support
If you encounter any issues or have questions, feel free to open an issue in this repository or contact me at minthukywe2020@gmail.com.

Enjoy exploring binary classification with Jupyter Notebook! 🚀
