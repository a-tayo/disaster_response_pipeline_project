# Disaster Response Pipeline Project

# Installation
The downloaded libraries used in this project are
```
Flask==2.1.3
joblib==1.1.0
nltk==3.7
numpy==1.23.1
pandas==1.4.3
plotly==5.9.0
scikit-learn==1.1.1
SQLAlchemy==1.4.39
```
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
The files and folders in this project is as structured below
   - app
   | - template
   | |- master.html  # main page of web app
   | |- go.html  # classification result page of web app
   |- run.py  # Flask file that runs app

   - data
   |- disaster_categories.csv  # data to process 
   |- disaster_messages.csv  # data to process
   |- process_data.py
   |- DisasterDatabase.db   # database to save clean data to

   - models
   |- train_classifier.py
   |- classifier.pkl  # saved model 

   - README.md

# Evaluation Report
Classification report for each category prediction
 ```
        Feature 1: related 
                      precision    recall  f1-score   support

                  0       0.74      0.27      0.39      1212
                  1       0.81      0.97      0.89      3994

            accuracy                           0.81      5206
          macro avg       0.78      0.62      0.64      5206
        weighted avg       0.80      0.81      0.77      5206


        Feature 2: request 
                      precision    recall  f1-score   support

                  0       0.90      0.99      0.94      4331
                  1       0.90      0.43      0.58       875

            accuracy                           0.90      5206
          macro avg       0.90      0.71      0.76      5206
        weighted avg       0.90      0.90      0.88      5206


        Feature 3: offer 
                      precision    recall  f1-score   support

                  0       1.00      1.00      1.00      5188
                  1       0.00      0.00      0.00        18

            accuracy                           1.00      5206
          macro avg       0.50      0.50      0.50      5206
        weighted avg       0.99      1.00      0.99      5206


        Feature 4: aid_related 
                      precision    recall  f1-score   support

                  0       0.76      0.89      0.82      3030
                  1       0.80      0.60      0.69      2176

            accuracy                           0.77      5206
          macro avg       0.78      0.75      0.75      5206
        weighted avg       0.77      0.77      0.76      5206


        Feature 5: medical_help 
                      precision    recall  f1-score   support

                  0       0.92      1.00      0.96      4793
                  1       0.48      0.03      0.05       413

            accuracy                           0.92      5206
          macro avg       0.70      0.51      0.50      5206
        weighted avg       0.89      0.92      0.89      5206


        Feature 6: medical_products 
                      precision    recall  f1-score   support

                  0       0.95      1.00      0.98      4946
                  1       0.89      0.07      0.12       260

            accuracy                           0.95      5206
          macro avg       0.92      0.53      0.55      5206
        weighted avg       0.95      0.95      0.93      5206


        Feature 7: search_and_rescue 
                      precision    recall  f1-score   support

                  0       0.98      1.00      0.99      5070
                  1       0.78      0.05      0.10       136

            accuracy                           0.97      5206
          macro avg       0.88      0.53      0.54      5206
        weighted avg       0.97      0.97      0.96      5206


        Feature 8: security 
                      precision    recall  f1-score   support

                  0       0.98      1.00      0.99      5125
                  1       0.00      0.00      0.00        81

            accuracy                           0.98      5206
          macro avg       0.49      0.50      0.50      5206
        weighted avg       0.97      0.98      0.98      5206


        Feature 9: military 
                      precision    recall  f1-score   support

                  0       0.97      1.00      0.98      5019
                  1       0.77      0.05      0.10       187

            accuracy                           0.97      5206
          macro avg       0.87      0.53      0.54      5206
        weighted avg       0.96      0.97      0.95      5206


        Feature 10: child_alone 
                      precision    recall  f1-score   support

                  0       1.00      1.00      1.00      5206

            accuracy                           1.00      5206
          macro avg       1.00      1.00      1.00      5206
        weighted avg       1.00      1.00      1.00      5206


        Feature 11: water 
                      precision    recall  f1-score   support

                  0       0.95      1.00      0.97      4859
                  1       0.88      0.21      0.34       347

            accuracy                           0.95      5206
          macro avg       0.91      0.60      0.66      5206
        weighted avg       0.94      0.95      0.93      5206


        Feature 12: food 
                      precision    recall  f1-score   support

                  0       0.92      0.99      0.96      4598
                  1       0.90      0.34      0.50       608

            accuracy                           0.92      5206
          macro avg       0.91      0.67      0.73      5206
        weighted avg       0.92      0.92      0.90      5206


        Feature 13: shelter 
                      precision    recall  f1-score   support

                  0       0.93      1.00      0.96      4757
                  1       0.91      0.19      0.32       449

            accuracy                           0.93      5206
          macro avg       0.92      0.60      0.64      5206
        weighted avg       0.93      0.93      0.91      5206


        Feature 14: clothing 
                      precision    recall  f1-score   support

                  0       0.98      1.00      0.99      5116
                  1       0.80      0.09      0.16        90

            accuracy                           0.98      5206
          macro avg       0.89      0.54      0.58      5206
        weighted avg       0.98      0.98      0.98      5206


        Feature 15: money 
                      precision    recall  f1-score   support

                  0       0.98      1.00      0.99      5088
                  1       0.88      0.06      0.11       118

            accuracy                           0.98      5206
          macro avg       0.93      0.53      0.55      5206
        weighted avg       0.98      0.98      0.97      5206


        Feature 16: missing_people 
                      precision    recall  f1-score   support

                  0       0.99      1.00      0.99      5150
                  1       1.00      0.02      0.04        56

            accuracy                           0.99      5206
          macro avg       0.99      0.51      0.51      5206
        weighted avg       0.99      0.99      0.98      5206


        Feature 17: refugees 
                      precision    recall  f1-score   support

                  0       0.97      1.00      0.98      5041
                  1       0.67      0.01      0.02       165

            accuracy                           0.97      5206
          macro avg       0.82      0.51      0.50      5206
        weighted avg       0.96      0.97      0.95      5206


        Feature 18: death 
                      precision    recall  f1-score   support

                  0       0.96      1.00      0.98      4987
                  1       0.83      0.09      0.16       219

            accuracy                           0.96      5206
          macro avg       0.90      0.55      0.57      5206
        weighted avg       0.96      0.96      0.95      5206


        Feature 19: other_aid 
                      precision    recall  f1-score   support

                  0       0.87      1.00      0.93      4522
                  1       0.76      0.02      0.04       684

            accuracy                           0.87      5206
          macro avg       0.82      0.51      0.48      5206
        weighted avg       0.86      0.87      0.81      5206


        Feature 20: infrastructure_related 
                      precision    recall  f1-score   support

                  0       0.94      1.00      0.97      4869
                  1       0.50      0.00      0.01       337

            accuracy                           0.94      5206
          macro avg       0.72      0.50      0.49      5206
        weighted avg       0.91      0.94      0.90      5206


        Feature 21: transport 
                      precision    recall  f1-score   support

                  0       0.96      1.00      0.98      4979
                  1       0.76      0.11      0.20       227

            accuracy                           0.96      5206
          macro avg       0.86      0.56      0.59      5206
        weighted avg       0.95      0.96      0.95      5206


        Feature 22: buildings 
                      precision    recall  f1-score   support

                  0       0.95      1.00      0.98      4949
                  1       0.81      0.07      0.12       257

            accuracy                           0.95      5206
          macro avg       0.88      0.53      0.55      5206
        weighted avg       0.95      0.95      0.93      5206


        Feature 23: electricity 
                      precision    recall  f1-score   support

                  0       0.98      1.00      0.99      5101
                  1       0.67      0.02      0.04       105

            accuracy                           0.98      5206
          macro avg       0.82      0.51      0.51      5206
        weighted avg       0.97      0.98      0.97      5206


        Feature 24: tools 
                      precision    recall  f1-score   support

                  0       1.00      1.00      1.00      5183
                  1       0.00      0.00      0.00        23

            accuracy                           1.00      5206
          macro avg       0.50      0.50      0.50      5206
        weighted avg       0.99      1.00      0.99      5206


        Feature 25: hospitals 
                      precision    recall  f1-score   support

                  0       0.99      1.00      0.99      5152
                  1       0.00      0.00      0.00        54

            accuracy                           0.99      5206
          macro avg       0.49      0.50      0.50      5206
        weighted avg       0.98      0.99      0.98      5206


        Feature 26: shops 
                      precision    recall  f1-score   support

                  0       1.00      1.00      1.00      5188
                  1       0.00      0.00      0.00        18

            accuracy                           1.00      5206
          macro avg       0.50      0.50      0.50      5206
        weighted avg       0.99      1.00      0.99      5206


        Feature 27: aid_centers 
                      precision    recall  f1-score   support

                  0       0.99      1.00      0.99      5148
                  1       0.00      0.00      0.00        58

            accuracy                           0.99      5206
          macro avg       0.49      0.50      0.50      5206
        weighted avg       0.98      0.99      0.98      5206


        Feature 28: other_infrastructure 
                      precision    recall  f1-score   support

                  0       0.95      1.00      0.98      4969
                  1       0.00      0.00      0.00       237

            accuracy                           0.95      5206
          macro avg       0.48      0.50      0.49      5206
        weighted avg       0.91      0.95      0.93      5206


        Feature 29: weather_related 
                      precision    recall  f1-score   support

                  0       0.86      0.97      0.91      3729
                  1       0.88      0.61      0.72      1477

            accuracy                           0.87      5206
          macro avg       0.87      0.79      0.81      5206
        weighted avg       0.87      0.87      0.86      5206


        Feature 30: floods 
                      precision    recall  f1-score   support

                  0       0.95      1.00      0.97      4767
                  1       0.91      0.37      0.53       439

            accuracy                           0.94      5206
          macro avg       0.93      0.68      0.75      5206
        weighted avg       0.94      0.94      0.93      5206


        Feature 31: storm 
                      precision    recall  f1-score   support

                  0       0.94      0.99      0.97      4733
                  1       0.81      0.38      0.51       473

            accuracy                           0.94      5206
          macro avg       0.88      0.68      0.74      5206
        weighted avg       0.93      0.94      0.92      5206


        Feature 32: fire 
                      precision    recall  f1-score   support

                  0       0.99      1.00      0.99      5144
                  1       0.00      0.00      0.00        62

            accuracy                           0.99      5206
          macro avg       0.49      0.50      0.50      5206
        weighted avg       0.98      0.99      0.98      5206


        Feature 33: earthquake 
                      precision    recall  f1-score   support

                  0       0.97      0.99      0.98      4713
                  1       0.90      0.71      0.80       493

            accuracy                           0.97      5206
          macro avg       0.94      0.85      0.89      5206
        weighted avg       0.96      0.97      0.96      5206


        Feature 34: cold 
                      precision    recall  f1-score   support

                  0       0.98      1.00      0.99      5099
                  1       0.80      0.04      0.07       107

            accuracy                           0.98      5206
          macro avg       0.89      0.52      0.53      5206
        weighted avg       0.98      0.98      0.97      5206


        Feature 35: other_weather 
                      precision    recall  f1-score   support

                  0       0.95      1.00      0.97      4921
                  1       1.00      0.01      0.02       285

            accuracy                           0.95      5206
          macro avg       0.97      0.51      0.50      5206
        weighted avg       0.95      0.95      0.92      5206


        Feature 36: direct_report 
                      precision    recall  f1-score   support

                  0       0.86      0.98      0.92      4205
                  1       0.83      0.35      0.49      1001

            accuracy                           0.86      5206
          macro avg       0.85      0.67      0.71      5206
        weighted avg       0.86      0.86      0.84      5206
```

# Model Accuracy 
   - Accuracy: 0.9458477397874248