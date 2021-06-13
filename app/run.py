import json
import plotly
import pandas as pd
import pickle

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
from sqlalchemy import create_engine
from collections import defaultdict
import re

app = Flask(__name__)

def tokenize(text):
    # Tokenize function used on input text messages

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('input_data', engine)

# load model

with open("../models/classifier.pkl", 'rb') as file:  
    model = pickle.load(file)

#model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    message_category_cols=list(df.columns)[4:]
    num_per_category=defaultdict(int)
    for category_name in message_category_cols:
        num_per_category[category_name]=df[category_name].sum()
           
    
    # create visuals
    graphs = [
        # {
        #     'data': [
        #         Bar(
        #             x=genre_names,
        #             y=genre_counts
        #         )
        #     ],

        #     'layout': {
        #         'title': 'Distribution of DB Message Genres',
        #         'yaxis': {
        #             'title': "Count"
        #         },
        #         'xaxis': {
        #             'title': "Genre"
        #         }
        #     }
        # },

        {
            'data': [
                Bar(
                    x=[*num_per_category],
                    y=[num_per_category[key] for key in num_per_category]
                )
            ],

            'layout': {
                'title': 'Distribution of DB Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True) # uncommet for local use


if __name__ == '__main__':
    main()