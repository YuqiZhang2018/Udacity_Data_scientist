import pandas as pd
from sqlalchemy import create_engine

# Load data
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories, messages, on='id')
    return df

# Clean data
def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] =  categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)
        
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df

# Save data
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('message_categories', engine, index=False)


def main():
    messages_filepath = "data/messages.csv"
    categories_filepath = "data/categories.csv"
    database_filepath = "data/InsertDatabaseName.db"
    
    print('Loading data...')
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print('Saving data...')
    save_data(df, database_filepath)

    print('Cleaned data saved to database!')
    

if __name__ == '__main__':
    main()