import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

# NLP pre-processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re

# model creation
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# model evaluation
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

      
def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('input_data',engine)
    df.head()
    X = df['message']
    columns = list(df.columns)
    Y = df[columns[4:]]
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    # initiate stop-words-BOW
    stop_words = stopwords.words("english")

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    #Regex to find urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Finds all urls from the provided text
    detected_urls = re.findall(url_regex, text)

    #Replaces all urls found with the "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    
    # tokenize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)

    # iterate through each token    
    tokens = [lemmatizer.lemmatize(word, pos='n') for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    
    return tokens


def build_model():
    # This machine pipeline should take in the message column 
    # as input and output classification results on the other
    # categories in the dataset.
    
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize)),
                     ('tfidf',TfidfTransformer()),
                     ('clf',MultiOutputClassifier(
                                    GradientBoostingClassifier(max_depth=9)))])
    parameters = {
    #     'vect__ngram_range': ((1, 1), (1, 2)),
    #     'vect__max_df': (0.5, 0.75, 1.0),
    #     'vect__max_features': (None, 5000, 10000),
    #     'tfidf__use_idf': (True, False),
    #     'clf__estimator__n_estimators': [50, 100, 200],
    #     'clf__estimator__min_samples_split': [2, 3, 4]
        'clf__estimator__max_depth' : [6,9]
        
    }

    model = GridSearchCV(pipeline, parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    # Evaluate model prediction score by printing each
    # attribute: Precision, Recall, F1-score
    Y_pred = model.predict(X_test)


    for idx,col in enumerate(category_names):
        results_dict=classification_report(
               Y_test.values[:,idx], Y_pred[:,idx],
               output_dict=True,zero_division=0)
        try:
            pre=results_dict['1']['precision']
            rec=results_dict['1']['recall']
            f1=results_dict['1']['f1-score']
        except:
            pre=results_dict['0']['precision']
            rec=results_dict['0']['recall']
            f1=results_dict['0']['f1-score']        
        print(f'Category name: {list(Y_test.columns)[idx]}\nPrecision: {pre:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}')


def save_model(model, model_filepath):
    # Save trained model as pickel file for future use

    with open(model_filepath, 'wb') as file:  
        pickle.dump(pipeline, file)


def main():
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