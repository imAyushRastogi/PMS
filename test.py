import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support as score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("predictive_maintenance.csv")

# Define the target variable
target_name = 'Failure Type'

# Print a summary of the train data
print(df.shape[0])
df.head(3)

# Create histograms for feature columns separated by the target column
def create_histogram(column_name):
    plt.figure(figsize=(16,6))
    return px.box(data_frame=df, y=column_name, color='Failure Type', points="all", width=1200)

create_histogram('air_temperature')
create_histogram('process_temperature')
create_histogram('rotational_speed')
create_histogram('torque')
create_histogram('tool_wear')

# Prepare the data for model training
def data_preparation(df_base, target_name):
    df = df_base.dropna()

    df['target_name_encoded'] = df[target_name].replace({'No Failure': 0, 'Power Failure': 1, 'Tool Wear Failure': 2, 'Overstrain Failure': 3, 'Random Failures': 4, 'Heat Dissipation Failure': 5})
    df['Type'] = df['Type'].replace({'L': 0, 'M': 1, 'H': 2})
    X = df.drop(columns=[target_name, 'target_name_encoded'])
    y = df['target_name_encoded'] # Prediction label

    # Split the data into x_train and y_train data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=0)

    # Print the shapes: the result is: (rows, training_sequence, features) (prediction value, )
    print('train: ', X_train.shape, y_train.shape)
    print('test: ', X_test.shape, y_test.shape)
    return X, y, X_train, X_test, y_train, y_test

# Remove target from training data
X, y, X_train, X_test, y_train, y_test = data_preparation(df, target_name)

# Train the XGBoost classification model
weight_train = compute_sample_weight('balanced', y_train)
weight_test = compute_sample_weight('balanced', y_test)

xgb_clf = XGBClassifier(booster='gbtree',  
                         tree_method='hist',  
                         eval_metric='aucpr',  
                         objective='multi:softmax',  
                         num_class=6)

# Fit the model to the data
xgb_clf.fit(X_train, y_train, sample_weight=weight_train)

# Make predictions on the test set
y_pred = xgb_clf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))