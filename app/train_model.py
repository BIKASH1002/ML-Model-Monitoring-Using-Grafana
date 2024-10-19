import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, confusion_matrix
from prometheus_client import Gauge, Histogram, Summary, start_http_server
import time
import warnings
warnings.filterwarnings("ignore")

# Prometheus Gauges for model metrics
accuracy_gauge = Gauge('loan_model_accuracy', 'Model Accuracy')
loss_gauge = Gauge('loan_model_loss', 'Model Loss')
precision_gauge = Gauge('loan_model_precision', 'Model Precision')
recall_gauge = Gauge('loan_model_recall', 'Model Recall')
f1_gauge = Gauge('loan_model_f1_score', 'Model F1 Score')

# Gauges for confusion matrix metrics
tp_gauge = Gauge('true_positive', 'True Positives')
tn_gauge = Gauge('true_negative', 'True Negatives')
fp_gauge = Gauge('false_positive', 'False Positives')
fn_gauge = Gauge('false_negative', 'False Negatives')

# Histogram for prediction distribution
prediction_histogram = Histogram('model_prediction_distribution', 'Prediction Value Distribution', buckets=[0, 0.25, 0.5, 0.75, 1])

# Summary for tracking training duration
training_time = Summary('model_training_duration_seconds', 'Time taken to train the model')

# Loading datasets
train_df = pd.read_csv('Data/train.csv')
test_df = pd.read_csv('Data/test.csv')

# Label Encoding for categorical columns
le = LabelEncoder()
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Handling outliers
train_df['person_age'] = train_df['person_age'].clip(upper=100)
test_df['person_age'] = test_df['person_age'].clip(upper=100)

train_df['person_emp_length'] = train_df['person_emp_length'].clip(upper=40)
test_df['person_emp_length'] = test_df['person_emp_length'].clip(upper=40)

X_train = train_df.drop(['id', 'loan_status'], axis=1)
y_train = train_df['loan_status']
X_test = test_df.drop(['id'], axis=1)

X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, 
                                                              test_size=0.2, random_state=42)

# Training RandomForest model and measure training time
@training_time.time()  # Measure training time with Prometheus
def train_model():
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_split, y_train_split)
    return rf

rf = train_model()

def log_metrics():
    y_pred = rf.predict(X_val)
    y_proba = rf.predict_proba(X_val)

    # Calculating metrics
    accuracy = accuracy_score(y_val, y_pred)
    loss = log_loss(y_val, y_proba)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

    # Log metrics to Prometheus
    accuracy_gauge.set(accuracy)
    loss_gauge.set(loss)
    precision_gauge.set(precision)
    recall_gauge.set(recall)
    f1_gauge.set(f1)
    tp_gauge.set(tp)
    tn_gauge.set(tn)
    fp_gauge.set(fp)
    fn_gauge.set(fn)

    for pred in y_pred:
        prediction_histogram.observe(pred)

    print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == '__main__':
    # Starting Prometheus server
    start_http_server(8000)
    print("Prometheus metrics server running on http://localhost:8000/metrics")

    while True:
        log_metrics()
        time.sleep(10)
