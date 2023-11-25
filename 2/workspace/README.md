# Disaster Response Pipeline

## Project Overview
This project aims to build a disaster response pipeline that classifies disaster-related messages using machine learning techniques. Through this system, messages can be quickly and accurately classified into the corresponding categories, allowing relevant agencies to better understand and respond to disaster situations.

The dataset comes from Figure Eight, which contains real-time disaster-related messages and corresponding classification labels. We will use this dataset for data processing, machine learning model training, and building a web application.

## How to Run the Python Scripts and Web Application
- Run the ETL script: `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
- Run the training script: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
- Run the web application: `python app/run.py`
- Access in the browser: [http://127.0.0.1:3000/](http://127.0.0.1:3000/)

## Repository File Descriptions
- `app/` directory: Contains the code and template files for the web application.
- `data/` directory: Contains the original dataset and the script for data processing.
- `models/` directory: Contains the script for training the classifier model and the saved model files.

## Data Processing
In the ETL script, we first load the original dataset, perform data cleaning and merging, and then store the processed data in a SQLite database.

## Machine Learning Process
In the script for training the classifier model, we used a custom tokenization function to process the text and built a machine learning pipeline, including feature extraction, model training, and evaluation.

## Web Application
The web application is built using the Flask framework. Users can enter messages and get corresponding classification results. The application also displays two visualizations based on the database to better describe the training data.

## Project Evaluation
We evaluated the performance of the model on the test set, including metrics such as F1 score, precision, and recall.

## Acknowledgments
Thanks to Figure Eight for providing the dataset and Udacity for providing project guidance.