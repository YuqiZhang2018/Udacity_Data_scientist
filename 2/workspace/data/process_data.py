#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[2]:


# import libraries
import pandas as pd

from sqlalchemy import create_engine


# In[11]:


def load_datasets(dataset_name):
    dataset = pd.read_csv(dataset_name)
    print(dataset.head())
    return dataset


# In[12]:


# load messages dataset and 
messages = load_datasets("messages.csv")


# In[13]:


# load categories dataset
categories = load_datasets("categories.csv")


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[20]:


def merge_dataset(dataset1,dataset2,connection):
    df = pd.merge(dataset1, dataset2, on=connection)
    print(df.head())
    df.to_csv('output.csv', index=False)
    return df


# In[21]:


df = merge_dataset(categories,messages,"id")


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[22]:


def split_columes(dataset,column,split_signal):
    # create a dataframe of the 36 individual category columns
    dataset = dataset[column].str.split(split_signal, expand=True)
    print(dataset.head())
    
    # select the first row of the categories dataframe    
    row = dataset.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    dataset_colnames = row.apply(lambda x: x[:-2])
    print(dataset_colnames)
    
    # rename the columns of `categories`
    dataset.columns = dataset_colnames
    dataset.head()
    return dataset


# In[25]:


categories = split_columes(df,"categories",";")


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[27]:


def convert_value(data):
    for column in data:
        # set each value to be the last character of the string
        data[column] =  data[column].astype(str).str[-1]

        # convert column from string to numeric
        data[column] = data[column].astype(int)
    print(data.head())
    return data


# In[28]:


categories = convert_value(categories)


# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.

# In[29]:


def categories_replace(dataset1,column, dataset2):
    # drop the original categories column from `df`
    dataset1.drop('categories', axis=1, inplace=True)
    print(dataset1.head())
    
    # concatenate the original dataframe with the new `categories` dataframe
    dataset1 = pd.concat([dataset1, dataset2], axis=1)
    dataset1.head()   
    
    return dataset1


# In[30]:


df = categories_replace(df, 'categories',categories)


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# In[33]:


def remove_duplicate(dataset):
    # check number of duplicates
    print(dataset.duplicated().sum())

    # drop duplicates
    dataset.drop_duplicates(inplace=True)

    # check number of duplicates
    print(dataset.duplicated().sum())
    
    return dataset


# In[34]:


df = remove_duplicate(df)


# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[36]:


engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName1', engine, index=False)


# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:




