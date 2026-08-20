import pandas as pd 
 
 
# Load the Titanic dataset 
titanic = pd.read_csv("../dataset/train.csv") 
 
# Display the first 5 rows 
print(titanic.head()) 
 
# Display dataset information 
print(titanic.info()) 
 
# Check missing values 
print(titanic.isnull().sum()) 
 
# Select the features 
X = titanic[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]] 
 
# Select the target variable 
y = titanic["Survived"] 
 
print("\nFeatures:") 
print(X.head()) 
 
print("\nTarget:") 
print(y.head()) 
 
# Handle missing Age values 
X["Age"] = X["Age"].fillna(X["Age"].median()) 
 
# Convert Sex into numerical values 
X["Sex"] = X["Sex"].map({ 
    "male": 0, 
    "female": 1 
}) 
 
print("\nFeatures after preprocessing:") 
print(X.head()) 
 
print("\nMissing values after preprocessing:") 
print(X.isnull().sum()) 
 
from sklearn.model_selection import train_test_split 
 
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split( 
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
) 
 
print("\nTraining data size:") 
print(X_train.shape) 
 
print("\nTesting data size:") 
print(X_test.shape) 
 
from sklearn.linear_model import LogisticRegression 
 
# Create the Logistic Regression model 
model = LogisticRegression(max_iter=1000) 
 
# Train the model using the training data 
model.fit(X_train, y_train) 
 
print("\nModel trained successfully!") 
 
# Make predictions on the test data 
y_pred = model.predict(X_test) 
 
print("\nPredictions:") 
print(y_pred[:10]) 
 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
 
# Calculate evaluation metrics 
accuracy = accuracy_score(y_test, y_pred) 
precision = precision_score(y_test, y_pred) 
recall = recall_score(y_test, y_pred) 
f1 = f1_score(y_test, y_pred) 
 
print("\n--- Model Evaluation ---") 
print("Accuracy :", accuracy) 
print("Precision:", precision) 
print("Recall   :", recall) 
print("F1 Score :", f1) 
 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt 
 
# Create the confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
 
print("\nConfusion Matrix:") 
print(cm) 
 
# Display the confusion matrix 
display = ConfusionMatrixDisplay( 
    confusion_matrix=cm, 
    display_labels=["Did Not Survive", "Survived"] 
) 
 
display.plot() 
 
plt.title("Confusion Matrix - Logistic Regression") 
plt.show() 
 
from sklearn.metrics import roc_curve, roc_auc_score 
 
# Get probability predictions for the positive class 
y_prob = model.predict_proba(X_test)[:, 1] 
 
# Calculate ROC curve 
fpr, tpr, thresholds = roc_curve(y_test, y_prob) 
 
# Calculate AUC 
auc = roc_auc_score(y_test, y_prob) 
 
print("\nAUC Score:", auc) 
 
# Plot ROC Curve 
plt.figure(figsize=(8, 6)) 
 
plt.plot( 
    fpr, 
    tpr, 
    label=f"Logistic Regression (AUC = {auc:.2f})" 
) 
 
# Random classifier reference line 
plt.plot( 
    [0, 1], 
    [0, 1], 
    linestyle="--", 
    label="Random Classifier" 
) 
 
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.title("ROC Curve - Logistic Regression") 
 
plt.legend() 
plt.show() 
 
# Create a copy of the test data for error analysis 
error_analysis = X_test.copy() 
 
# Add actual and predicted values 
error_analysis["Actual"] = y_test 
error_analysis["Predicted"] = y_pred 
 
# Find incorrect predictions 
errors = error_analysis[ 
    error_analysis["Actual"] != error_analysis["Predicted"] 
] 
 
print("\nNumber of incorrect predictions:", len(errors)) 
 
print("\nIncorrect Predictions:") 
print(errors.head(10)) 
 
# False Positives 
false_positives = errors[ 
    (errors["Actual"] == 0) & 
    (errors["Predicted"] == 1) 
] 
 
# False Negatives 
false_negatives = errors[ 
    (errors["Actual"] == 1) & 
    (errors["Predicted"] == 0) 
] 
 
print("\nFalse Positives:", len(false_positives)) 
print("False Negatives:", len(false_negatives)) 
 
# -------------------------------- 
# Improved Model 
# -------------------------------- 
 
# Create a new feature set including Embarked 
X_improved = titanic[ 
    ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"] 
].copy() 
 
# Fill missing Age values 
X_improved["Age"] = X_improved["Age"].fillna( 
    X_improved["Age"].median() 
) 
 
# Fill missing Embarked values 
X_improved["Embarked"] = X_improved["Embarked"].fillna( 
    X_improved["Embarked"].mode()[0] 
) 
 
# Convert Sex into numerical values 
X_improved["Sex"] = X_improved["Sex"].map({ 
    "male": 0, 
    "female": 1 
}) 
 
# Convert Embarked into numerical values 
X_improved["Embarked"] = X_improved["Embarked"].map({ 
    "S": 0, 
    "C": 1, 
    "Q": 2 
}) 
 
print("\nImproved Features:") 
print(X_improved.head()) 
 
print("\nMissing values:") 
print(X_improved.isnull().sum()) 
 
# Split the improved data 
X_train_imp, X_test_imp, y_train_imp, y_test_imp = train_test_split( 
    X_improved, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y 
) 
 
# Create the improved Logistic Regression model 
improved_model = LogisticRegression(max_iter=1000) 
 
# Train the improved model 
improved_model.fit(X_train_imp, y_train_imp) 
 
# Make predictions 
y_pred_imp = improved_model.predict(X_test_imp) 
 
print("\nImproved model trained successfully!") 
 
# Calculate improved model metrics 
 
accuracy_imp = accuracy_score(y_test_imp, y_pred_imp) 
precision_imp = precision_score(y_test_imp, y_pred_imp) 
recall_imp = recall_score(y_test_imp, y_pred_imp) 
f1_imp = f1_score(y_test_imp, y_pred_imp) 
 
# Calculate AUC 
y_prob_imp = improved_model.predict_proba(X_test_imp)[:, 1] 
auc_imp = roc_auc_score(y_test_imp, y_prob_imp) 
 
print("\n--- Improved Model Evaluation ---") 
print("Accuracy :", accuracy_imp) 
print("Precision:", precision_imp) 
print("Recall   :", recall_imp) 
print("F1 Score :", f1_imp) 
print("AUC      :", auc_imp) 
 
import numpy as np 
import matplotlib.pyplot as plt 
 
# Model performance values 
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"] 
 
original_scores = [ 
    accuracy, 
    precision, 
    recall, 
    f1, 
    auc 
] 
 
improved_scores = [ 
    accuracy_imp, 
    precision_imp, 
    recall_imp, 
    f1_imp, 
    auc_imp 
] 
 
# Set bar positions 
x = np.arange(len(metrics)) 
width = 0.35 
 
# Create the chart 
plt.figure(figsize=(10, 6)) 
 
plt.bar( 
    x - width / 2, 
    original_scores, 
    width, 
    label="Original Model" 
) 
 
plt.bar( 
    x + width / 2, 
    improved_scores, 
    width, 
    label="Improved Model" 
) 
 
plt.xlabel("Performance Metrics") 
plt.ylabel("Score") 
plt.title("Original vs Improved Logistic Regression Model") 
 
plt.xticks(x, metrics) 
plt.ylim(0, 1) 
 
plt.legend() 
 
plt.tight_layout() 
plt.show() , this is the full code , just say give what to upload
