# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])

# import libraries
import sys
import re
import joblib
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import (CountVectorizer, 
                                             TfidfTransformer)

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def load_data(database_filepath):
    """
    Loads the cleaned data from the database into memory in a pandas dataframe

    Parameters
    ----------
    database_filepath(str): The path to the database file 
    
    Return
    ------
    X(numpy Array(str)): Array of strings of messages
    Y(numpy Array(int)): Array of message categories 36 in total
    category_names: Array of all categories labels 
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', engine)

    # split into predictor and response variables
    X = df.loc[:,'message'].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns

    return X, Y, category_names

def tokenize(text):
    """
    Tokenizes a text into tokens

    Parameters
    ----------
    text(str): The text to tokenize
    
    Returns
    -------
    tokens(list): List of cleaned message tokens  
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Builds the model and fine tune it with a grid search

    Returns
    -------
    model(Model): The model object
    """
    # initialize a random forest classifier
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    # make a data processing pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(rf, n_jobs=-1) )
    ])

    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 50, 60]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, cv=2)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model on the test set

    Parameters
    ----------
    model(Model): The model object
    X_test(numpy.ndarray): The training dataset
    Y_test(numpy Array): The expected response
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print()
        print(f'Feature {i+1}:',category_names[i],'\n',classification_report(Y_test[:,i], Y_pred[:,i]))
    print('Accuracy', (Y_test == Y_pred).mean().mean())


def save_model(model, model_filepath):
    """
    Saves the model to a file

    Parameters
    ----------
    model(Model): The model object
    model_filepath: str
    """
    joblib.dump(model, model_filepath)


def main():
    """
    Main function
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()