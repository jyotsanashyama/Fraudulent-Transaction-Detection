# Fraudulent-Transaction-Detection

This project aims to develop a machine learning model to detect fraudulent transactions for a financial company. Logistic Regression was chosen as the model for classification, and the results were fine-tuned using GridSearchCV. Imbalanced data was handled using SMOTE, and a smaller subset of the resampled data was used to optimize computational efficiency.

# DATASET

The dataset contains 6,362,620 rows and 10 columns of transaction data.
[Download the Dataset](https://drive.google.com/file/d/1SiPNbeN3Bj7S4nnz2ov2Fv64Nv5W9dtb/view?usp=sharing)  

Key columns in the dataset are:  
1. type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.  
2. isFraud - Indicates whether the tracsaction is fraudulent (1 = Fraud, 0 = Not Fraud).  
3. amount - amount of the transaction in local currency.  

# EXPLORATORY DATA ANALYSIS (EDA)

To understand the distribution of transaction types and fraud occurrences, the following steps were performed:  
1. Checked for null values.  
2. Plotted count plot for column - 'type' and 'isFraud'.  
   
Observation is :  
a). for type columns:   
-> CASH_OUT has the highest number of transactions.  
-> DEBIT has the lowest number of transactions.  
b). for isFraud: Fraudulent transactions are greater than non-fraudulent transactions.  

# DATA PREPROCESSING

1. Handling Categorical Data:  
   -> The 'type' column was encoded using One-Hot Encoding.  
2. Handling Imbalanced Data:  
   -> Since the dataset is highly imbalanced, SMOTE (Synthetic Minority Over-sampling     Technique) was applied to balance the classes.  
3. Sampling a Subset of Data:  
   -> To make the process more efficient, a 10% subset of the resampled data was used for model training and evaluation.  
4. Splitting the Data:  
   -> The data was split into 80% for training and 20% testing sets.  
5. Scaling the Features:  
   -> Features were scaled using StandardScaler to normalize the data for optimal model performance.  

# MODEL TRAINING

A Logistic Regression model was trained using the resampled and scaled dataset. The initial model achieved an accuracy of 96.35% on the test data.  
-> Confusion Matrix was created.  
-> Classification Report: Precision, Recall, and F1-Score were calculated.  

# Hyperparameter Tuning

To fine-tune the Logistic Regression model, GridSearchCV was applied to search for the best hyperparameters. The grid search was conducted on parameters such as C (regularization strength) and solver.
