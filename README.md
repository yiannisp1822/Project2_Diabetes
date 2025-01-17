# Project2_Diabetes

**Predicting Diabetes in Patients: A Machine Learning Model**  

A Machine Learning Project by Yiannis Pagkalos, Sami Chowdhury, Mei Kam Bharadwaj, Dhwani Patel, and Lauren Christiansen  

**Executive Summary**  

Our project goal is to determine how key datapoints (BMI, high blood pressure, cholesterol, stroke, heart disease/attack, physical activity level, general health level, physical health level, difficulty walking scale, age, education level, income level) relate to the diagnosis of diabetes in different patients. We used this underlying data and created & trained a machine learning model to easily predict whether a patient would be diagnosed.


**Installation & Usage**  

Prerequisites:

Python 3.x
pandas
Matplotlib
Seaborn
Pydotplus
xgboost
Jupyter Notebook
Git version control system
Internet connection for data downloads


Setup:  

Clone the repository:
https://github.com/yiannisp1822/Project2_Diabetes.git

Install required packages: 

Launch Jupyter Notebook

Download required datasets:
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data
Place in Resources/ directory
Verify file integrity using provided checksums
Analysis Results
Machine Learning Methods

Our modelling employed multiple approaches:
Random Forest
Logistic Regression
KNN
Decision Tree
XG Boost
Confusion Matrix
Grid Search
Machine Learning Findings

We discovered just slight variations between models and their accuracy scores:


Model
Accuracy score
random forest
0.72
grid search - k means
0.7353
k means
0.74
grid search - logistic regression
0.75
logistic regression
0.751
grid search - random forest
0.753
random forest
0.7536
grid search - XGB classifier
0.7555
xg boost
0.756



Key Visualizations
We want to highlight these detailed visualizations:
Decision Tree 



Random Forest - Grid Search
   


Logistic Regression - Grid Search



KNN - Grid Search
   


**Project Evolution: Data Pivot**  

Initial Direction
Originally focused on datasets with less volume but including valuable datapoints such as H1BC 
https://www.kaggle.com/code/chanchal24/diabetes-dataset-eda-prediction-with-7-models
https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset


Challenges Encountered:
Potential of false negatives and false positives given modeling accuracy scores
Qualitative vs quantitative data challenges
Potential dangers of self-reported data

Data accuracy - where/how was it collected?
Data Collection & Preprocessing
Initial Assessment
Data source verification
Quality validation
Format standardization
Missing value analysis
Data Cleaning & Integration
Post-cleaning validation
Category Grouping
Model Methodology
Running various models trying to get varied accuracy scores

Limitations and Considerations:
Data Limitations
Different methodologies (self-reported data)
Methodological Constraints
Healthcare infrastructure differences
Socioeconomic factors
External Factors
Healthcare system adaptations

**Summary of Findings:**
Key Insights
After evaluating the initial results, we focused on testing models on the data after removing low correlation features (<0.1) and encoding only the BMI features (as it was the non-numeric one).


Given the nonlinear classification of the data, we experimented with models:
KNeighborsClassifier
Logistic Regression
RandomForestClassifier
XGBClassifier

To improve initial results, we ran GridSearchCV on each model to ensure we are using each model’s parameters optimally.

**POTENTIAL NEXT STEPS:**
We would explore more datasets that include health indicators such as HbA1C (hemoglobin A1C) and fast blood sugar test (FBS). 

HbA1C test measures the average blood sugar (glucose) level over the past 60-90 days.
 
A fasting  blood sugar test measures the blood sugar levels first thing in the morning before the patient breaks their fast. If the patient’s blood sugar is high, then it indicates that patient has difficulties breaking down sugar in their body. 

It is best to look at a dataset that includes both HbA1C and FBS data. HbA1C tests are less sensitive compared to the FBS test, but provides a more comprehensive story on the patient’s blood sugar over a period of months. 

In practice, both tests are used in the office to get a more accurate diagnosis of diabetes.


**Contributors**
Yiannis Pagkalos: Project Lead
Sami Chowdhury: Data Sourcing, Data Cleaning, Methodology
Mei Kam Bharadwaj: Documentation, Data Modeling
Dhwani Patel: Data Visualization
Lauren Christiansen: Data Visualization, Data Grouping


**Acknowledgments**
Data providers: Kaggle
Academic advisors
Healthcare professionals who provided data for analysis


