import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load message and categories datasets, merge data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the messages and categories datasets using the common id
    # Assign this combined dataset to df, which will be cleaned in the following steps
    df = messages.merge(categories, on='id',how='inner')
    return df


def clean_data(df):
    # Split the values in the categories column on the ; character so that each value
    # Use the first row of categories dataframe to create column names for the categories data.
    # Rename columns of categories with new column names.    
    categories = df['categories'].str.split(';', expand=True)   

    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    row = categories.iloc[0]   
    category_colnames = row.str[:-2].values
    
    # rename the columns of `categories`
    categories.columns = category_colnames   

    # Iterate through the category columns in df to keep only the last character of each string 
    # (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1].astype(int)

    # we can see that columns 'related' has a value of 2, we will convert it to 1
    # we can see that columns 'child_alone' has only a single value, we will drop it
    categories = categories.drop(['child_alone'],axis=1)
    categories = categories.replace(2, 1)

    # Drop the categories column from the df dataframe since it is no longer needed.
    # Concatenate df and categories data frames.
    df = df.drop(['categories'],axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop the duplicates.
    df.drop_duplicates(inplace=True)
    
    return df



def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)  
    df.to_sql('input_data', engine, index=False,if_exists='replace')

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