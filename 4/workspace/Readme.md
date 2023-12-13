# Udacity Course - Data scientist: Starbucks Capstone Challenge

## Project Overview

This project aims to analyze Starbucks' coupon distribution events to identify the effectiveness of coupons for different user groups. Using data analysis on user demographics, transaction records, and coupon details, we strive to develop predictive outcomes for coupon redemption. Our goal is to optimize coupon distribution strategies to enhance customer engagement and maximize marketing ROI.

Udacity provides technical and data support for this project.

## Dataset

The analysis is based on data from three JSON files:

- `portfolio.json`: Offer IDs and metadata (duration, type, etc.)
- `profile.json`: Demographic data for each customer
- `transcript.json`: Records for transactions, offers received, offers viewed, and offers completed

### Schema and Variables

#### `portfolio.json`

- `id`: Offer ID (string)
- `offer_type`: Type of offer (e.g., BOGO, discount, informational) (string)
- `difficulty`: Minimum spend to complete an offer (int)
- `reward`: Reward for completing an offer (int)
- `duration`: Time the offer is active (days) (int)
- `channels`: Delivery channels (list of strings)

#### `profile.json`

- `age`: Age of the customer (int)
- `became_member_on`: Date of app account creation (int)
- `gender`: Gender of the customer ('M', 'F', or 'O' for other) (str)
- `id`: Customer ID (str)
- `income`: Customer's income (float)

#### `transcript.json`

- `event`: Record description (transaction, offer received, etc.) (str)
- `person`: Customer ID (str)
- `time`: Time in hours since start of test (int)
- `value`: Offer ID or transaction amount (dict of strings)

## Problem Statement

The challenge is optimizing coupon distribution to attract consumer spending while minimizing the costs of marketing and discounts. Current coupon policies based on random distribution or simple spend-based distribution are not yielding the best results. This project aims to predict the effectiveness of giving specified coupons to specified users by considering user characteristics, coupon diversity, and transactional factors.

## Summary of Results

#### Model Selection Comparison Process

When choosing a model for the business objective of stimulating consumption using coupons, our primary focus is on the recall metric for label 1, which represents the successful use of a coupon. The reason for this focus is to maximize the identification of all potential customers who would respond positively to the coupon incentive, thus reducing the chance of missing out on potential sales.

We have three candidate models, each with its classification report and confusion matrix. Here's a brief comparison:

MLPClassifier shows the highest overall accuracy at 69% and shares the highest F1-score for label 1 with the RandomForestClassifier. Its precision and recall are balanced for both classes.

DecisionTreeClassifier has a slightly lower accuracy at 68% but stands out with the highest recall for label 1 at 72%. This indicates it is the best at identifying customers who will use the coupon, albeit at the expense of a higher false positive rate.

RandomForestClassifier also has an accuracy of 68% but has a lower recall for label 1 compared to the DecisionTreeClassifier. It provides a balanced approach but does not excel in the recall for label 1.

#### Model Conclusion

Given the business requirement is to maximize the identification of customers who will use coupons (label 1), the DecisionTreeClassifier is the recommended model. Its recall of 72% for label 1 is the highest among the models, suggesting that it will be the most effective at capturing potential coupon users. While this may result in a higher number of false positives (customers who won't use the coupon but are predicted to do so), this is acceptable within our strategic framework, as the cost of misidentifying non-responsive customers is likely lower than the opportunity cost of missing out on responsive ones.

## Library used
- import pandas
- import numpy
- import math
- import json
- import pandas
- import matplotlib.pyplot
- import seaborn
- from tqdm.auto import tqdm
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.metrics import classification_report, accuracy_score
- from sklearn.preprocessing import OneHotEncoder
- from sklearn.neural_network import MLPClassifier
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.svm import SVC
- from sklearn.model_selection import GridSearchCV

## Acknowledgements

Thanks for Udacity.

My BLOG is https://medium.com/@yuqi.zhang2020/udacity-course-data-scientist-starbucks-capstone-challenge-fd8e4db36fc7