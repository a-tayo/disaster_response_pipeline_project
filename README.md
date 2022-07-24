# Disaster Response Pipeline Project

# Installation
The downloaded libraries used in this project are
'''
Flask==2.1.3
joblib==1.1.0
nltk==3.7
numpy==1.23.1
pandas==1.4.3
plotly==5.9.0
scikit-learn==1.1.1
SQLAlchemy==1.4.39
'''
All other libraries are available in the Anaconda Distribution of python
### Instructions:
1. Run the following commands in the project's root directory to set up your  environment, database and model.

    - Create a python virtual environment with conda
      `conda -n create <env_name> python=3.10.5`
    - Switch to the new environment
      `conda activate <env_name>`
    - Install the requirements
      `pip install -r requirements.txt`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Project Motivation
### Problem:
    - Disaster response teams finds it difficult to filter out the relevant information from all the communications received following a disaster.
### Solution: 
    - Machine learning model connected to a web app to categorise messages and filter out the relevant ones.

# Project Description
This project is based on the dataset curated by [Appen](https://appen.com)(formerly known as figure eight) and provided by Udacity. The project is divided into three main parts

  1. ETL(Extract Transform Load):
        At this stage, the data is loaded from the csv files provided and necessary cleaning and transformation operation is performed on it then the cleaned and transformed dataset set is saved in a sqlite database.
  
  2. Machine Learning:
        At this stage, the cleaned data is loaded into memory and used to train and evaluate a supervised machine learning model which will perform the classification task. 
        
        The resulting model is then saved to a file `classifier.pkl` which will be loaded at the final stage which is the web app to make classifications of disaster messages on demand.

  3. Flask Web App:
        The web app is the interface the user will interact with and it provides the following functionality.
        - A form for the user to enter a disaster message to be classified.
        - Outputs list and higlights the predicted category of the input message.
        - Shows a visualization of the original dataset values.
  
# Files Description
    