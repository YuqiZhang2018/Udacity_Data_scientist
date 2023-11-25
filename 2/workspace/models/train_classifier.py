#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
from sqlalchemy import create_engine
import pandas as pd


# In[2]:


# load data from database

engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table("InsertTableName", con=engine)


# In[3]:


print(df.columns)


# In[4]:


df.head(2)


# In[5]:


X = df['message']
Y = df.iloc[:, 4:]


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

# In[9]:


# Split data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Train classifier
pipeline.fit(X_train, Y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[10]:


# Predict on test data

Y_pred = pipeline.predict(X_test)


# In[11]:


from sklearn.metrics import classification_report

# Get a list of the column names of Y for iteration
target_names = Y.columns.tolist()

# Convert Y_test and Y_pred into DataFrames for easier manipulation
Y_test_df = pd.DataFrame(Y_test, columns=target_names)
Y_pred_df = pd.DataFrame(Y_pred, columns=target_names)

for column in target_names:
    print('------------------------------------------------------\n')
    print('FEATURE: {}\n'.format(column))
    print(classification_report(Y_test_df[column], Y_pred_df[column]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[12]:


pipeline.get_params()


# In[21]:


from sklearn.model_selection import GridSearchCV

parameters = {
    'clf__estimator__n_estimators' : [50, 100],
    'clf__estimator__min_samples_split' : [2, 5],
    'clf__estimator__min_samples_leaf' : [2, 4],
}

cv = GridSearchCV(pipeline, param_grid=parameters)


# In[22]:


cv


# In[23]:


cv.fit(X_train, Y_train)


# In[24]:


cv.best_params_


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[25]:


Y_pred = pipeline.predict(X_test)

# Get a list of the column names of Y for iteration
target_names = Y.columns.tolist()

# Convert Y_test and Y_pred into DataFrames for easier manipulation
Y_test_df = pd.DataFrame(Y_test, columns=target_names)
Y_pred_df = pd.DataFrame(Y_pred, columns=target_names)

for column in target_names:
    print('------------------------------------------------------\n')
    print('FEATURE: {}\n'.format(column))
    print(classification_report(Y_test_df[column], Y_pred_df[column]))


# In[27]:


accuracy = (Y_pred == Y_test).mean()
accuracy


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




