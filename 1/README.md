# FordGoBike Data Analysis

## Libraries Used
The following Python libraries are used for this analysis:

- pandas: For data manipulation and analysis.
- numpy: For numerical operations.
- sklearn: For machine learning models and metrics.
- seaborn: For data visualization.
- matplotlib: For creating static, animated, and interactive visualizations.
- folium: For creating maps.
- collections: For creating defaultdict.

## Project Motivation
This project utilizes the "fordgobike" dataset which records each share bike usage. The objective is to answer the following business questions:

1. Analyze the relationship between the duration of usage and the type of user (subscriber or customer).
2. Analyze the optimal placement of shared bicycles using clustering methods.
3. Analyze the temporal periodicity of usage to understand user behavior.

## Repository Contents
The repository contains two main files:

1. `201902-fordgobike-tripdata.csv`: This is the dataset that records the usage of fordgobike shared bicycles over a month.
2. `program 1.ipynb`: This contains the data analysis code.

## File Descriptions
1. `201902-fordgobike-tripdata.csv`: This CSV file contains records of fordgobike shared bicycle usage over a month. It includes details like trip duration, start time, end time, user type, etc.
2. `program 1.ipynb`: This Jupyter notebook file contains all the Python code used for data cleaning, exploration, analysis, and visualization. It is well-documented and explains the process and results.

## Summary of Results
1. Impact of User Type: The "Subscriber" type of users have a significantly higher usage count than "Customer" type of users. This suggests that "Subscriber" type users are long-term users who may use the service more frequently in their daily lives, while "Customer" type users may be temporary users or tourists who use the service less frequently.
2. Difference in Duration: The duration of use by "Customer" type users is generally longer than that of "Subscriber" type users. This might indicate that "Customer" users are more likely to be using the service for leisure activities, while "Subscriber" users are more likely to be commuting short distances.
3. Usage throughout the Week: The usage count is generally lower on weekends (Saturday and Sunday) than on weekdays. However, the average usage duration is longer on weekends. This suggests that users are more likely to use the service for leisure rather than commuting on the weekends.

## Acknowledgements
Thanks to [FordGoBike](https://www.fordgobike.com/) for making this dataset publicly available for use. This project would not have been possible without this dataset. Also, thanks to all the Python and Data Science community for their valuable resources and help.