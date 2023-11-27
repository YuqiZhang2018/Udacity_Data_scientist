import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import nltk


# loads the data from the database.
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("InsertTableName", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y

# tokenizes the text
def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

# builds a machine learning pipeline and uses GridSearchCV for hyperparameter tuning
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators' : [50, 100],
        'clf__estimator__min_samples_split' : [2, 5],
        'clf__estimator__min_samples_leaf' : [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

# evaluates the model and prints the classification report
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    target_names = Y.columns.tolist()
    Y_test_df = pd.DataFrame(Y_test, columns=target_names)
    Y_pred_df = pd.DataFrame(Y_pred, columns=target_names)

    for column in target_names:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test_df[column], Y_pred_df[column]))

# saves the trained model as a pickle file
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    X, Y = load_data('data/InsertDatabaseName.db')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    model = build_model()
    model.fit(X_train, Y_train)
    evaluate_model(model, X_test, Y_test)
    save_model(model, 'model.pkl')
    
if __name__ == "__main__":
    main()