# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Read in two csv files into dataframes, merges them 
       and return a single dataframe
    
    Parameters
    ----------- 
    messages_filepath(str): 
            filepath to the csv file containing messages
           
    categories_filepath(str):
            filepath to the csv file containing the categories

    Returns
    --------
    Merged Dataframe containing both the categories and the messages 
    """
    # read in the data to a pandas dataframe
    messages =pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the cateogries and messages dataframe on common id
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Takes a dataframe and performs cleaning operation
        on it

    Parameters
    -----------
    df:
        dataframe to clean
    
    Return
    ------
        cleaned dataframe
    """

    categories = df.categories.str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # extract a list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert categories 
    categories = categories.applymap(lambda x:int(x[-1:]))

    # drop the rows from which have a value of 2 for the related cateogry
    categories.drop(categories[categories.iloc[:,0]==2].index, inplace=True)

    # drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, right_index=True, left_index=True)

    # drop duplicates
    df.drop(df[df.duplicated()].index, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Saves the into a sqlite database file

    Parameters
    ----------
    df:
        dataframe to save
    database_filename:
        filename to save the data in"
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()