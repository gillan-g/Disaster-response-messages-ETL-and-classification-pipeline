# Disaster Response Pipeline Project
### Link to web app:

### Description
During a disaster the need for fast and accurate response is essential to save lifes.
A pain during such unfotunate incidences is the need to process high volume of information passed through social media, 
filter relevant information and delegate the information to relevant teams.

This application is a dummy attempt to create a solution for the above. 

The empty field accepts a text message, classifies it as relevant to a disaster or not, and outputs to the user the message tag.   

### Dependencies



### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py ..data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### File Descriptions
.
├── README.md
├── app : Flask App Files
│ ├── run.py : Flask file to  run the app
│ └── templates
│ ├── go.html
│ └── master.html
├── data : It contains all ETL Files 
│ ├── DisasterResponse.db :  SQLite DataBase file containing cleaned data after ETL process
│ ├── disaster_categories.csv :  Disaster Categories CSV file
│ ├── disaster_messages.csv : Messages CSV file
│ └── process_data.py : ETL pipeline code
├── models : It contains all ML files
│ ├── classifier.pkl : classifier produced by train_classifier file
│ └── train_classifier.py : ML pipeline classification code

### Data flow overview
This project performs the following steps:
ETL pipeline:
- Read csv files as input
- Transform and clean data into useable format
- Load tables to SQLite DB

NLP ML pipeline
- Load input data from SQLite DB
- Tokenize text messages and use Tfidf transformation
- Split into train and test datasets 
- Apply MultiOutputClassifier on all classification targets using GradientBoostingClassifier  
- Optimize parameters using GridsearchCV
- Evaluate model classifiaction score
- Save model as a pickel file for future use

Finally, the application called using Flask framework.


### Licensing, Acknowledgements
Data is courtesy of Figure-8, recently purchased by Appen https://appen.com/