#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[2]:


# import libraries
from sqlalchemy import create_engine
import pandas as pd


# In[3]:


def load_data(database_name, table_name):
    engine = create_engine(f'sqlite:///{database_name}.db')

    df = pd.read_sql_table(table_name, con=engine)
   
    print(df.columns)
    print(df.head(2))

    X = df['message']
    Y = df.iloc[:, 4:]

    return X, Y

X, Y = load_data('InsertDatabaseName', 'InsertTableName1')


# ### 2. Write a tokenization function to process your text data

# In[6]:


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def tokenize(text):
    """
    Tokenize and clean text by breaking it into words, lemmatizing, 
    converting to lower case and removing leading/trailing white space.

    Parameters:
    text (str): Text to be tokenized.

    Returns:
    list: List of clean, lemmatized tokens.
    """
    # Initialize WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Tokenize text into words
    tokens = word_tokenize(text)

    # Initialize an empty list to hold the cleaned tokens
    clean_tokens = []

    # Iterate over each token
    for token in tokens:
        # Lemmatize, convert to lower case and remove leading/trailing white space
        clean_token = lemmatizer.lemmatize(token).lower().strip()

        # Add the clean token to the list
        clean_tokens.append(clean_token)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import nltk

# Create a machine learning pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[13]:


from sklearn.model_selection import train_test_split

def train_classifier(X, Y, pipeline, test_size=0.2, random_state=42):
    """
    Train a classifier using a training pipeline.

    Parameters:
    X: Features dataset.
    Y: Labels dataset.
    pipeline: The machine learning pipeline that includes the preprocessing and the classifier.
    test_size: The proportion of the dataset to include in the test split (default is 0.2).
    random_state: The seed used by the random number generator (default is 42).

    Returns:
    The trained pipeline.
    """
    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Train classifier
    pipeline.fit(X_train, Y_train)

    # Return the trained pipeline and the test set for further evaluation
    return pipeline, X_train, X_test, Y_train, Y_test

trained_pipeline, X_train, X_test, Y_train, Y_test = train_classifier(X, Y, pipeline)


# In[9]:


# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Train classifier
pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(pipeline, X_test, Y_test):
    """
    Predict on test data and print classification report for each label feature.

    Parameters:
    pipeline: The trained machine learning pipeline.
    X_test: Test dataset features.
    Y_test: Test dataset labels.

    Returns:
    Y_pred_df: DataFrame containing the predictions.
    """
    # Predict on test data using the pipeline
    Y_pred = pipeline.predict(X_test)

    # Get a list of the column names of Y for iteration
    target_names = Y_test.columns.tolist()

    # Convert Y_test and Y_pred into DataFrames for easier manipulation
    Y_test_df = pd.DataFrame(Y_test, columns=target_names)
    Y_pred_df = pd.DataFrame(Y_pred, columns=target_names)

    # Print classification report for each feature
    for column in target_names:
        print('------------------------------------------------------\n')
        print(f'FEATURE: {column}\n')
        print(classification_report(Y_test_df[column], Y_pred_df[column]))

    return Y_pred_df

Y_pred_df = evaluate_model(trained_pipeline, X_test, Y_test)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[15]:


from sklearn.model_selection import GridSearchCV

def perform_grid_search(pipeline, X_train, Y_train):
    """
    Perform grid search to find the best parameters for the pipeline.

    Parameters:
    pipeline: The machine learning pipeline on which to perform grid search.
    X_train: Training dataset features.
    Y_train: Training dataset labels.

    Returns:
    cv: The fitted GridSearchCV object.
    """
    # Define the parameter grid to search
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 5],
        'clf__estimator__min_samples_leaf': [2, 4],
    }

    # Create GridSearchCV object with the pipeline, parameter grid, and verbose output
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    # Fit GridSearchCV
    cv.fit(X_train, Y_train)

    # Return the GridSearchCV object to access the results
    return cv


cv = perform_grid_search(pipeline, X_train, Y_train)
print(cv.best_params_)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[16]:


import pandas as pd
from sklearn.metrics import classification_report

def predict_and_evaluate(pipeline, X_test, Y_test):
    """
    Predict on test data, print classification report for each label, and calculate accuracy.

    Parameters:
    pipeline: The trained machine learning pipeline.
    X_test: Test dataset features.
    Y_test: Test dataset labels.

    Returns:
    accuracy: The mean accuracy across all label predictions.
    """
    # Predict on test data using the pipeline
    Y_pred = pipeline.predict(X_test)

    # Get a list of the column names of Y for iteration
    target_names = Y_test.columns.tolist()

    # Convert Y_test and Y_pred into DataFrames for easier manipulation
    Y_test_df = pd.DataFrame(Y_test, columns=target_names)
    Y_pred_df = pd.DataFrame(Y_pred, columns=target_names)

    # Print classification report for each feature
    for column in target_names:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test_df[column], Y_pred_df[column]))

    # Calculate and return the mean accuracy across all label predictions
    accuracy = (Y_pred_df == Y_test_df).mean()
    return accuracy


accuracy = predict_and_evaluate(pipeline, X_test, Y_test)
print('Mean Accuracy:', accuracy)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:





# In[ ]:





# ### 9. Export your model as a pickle file

# In[28]:


import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(cv, file)
    


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




