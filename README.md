# Mental Health Treatment Prediction in the Tech Industry

This project aims to predict whether a person working in tech will seek mental health treatment based on survey data
from [OSMI Mental Health in Tech Survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey).

## Project Overview
Mental health is an increasingly critical topic, especially in high-pressure environments like tech. This machine
learning project seeks to classify individuals based on their likelihood of seeking treatment for mental health issues,
using survey data that captures demographics, workplace support, and mental health history.


## Objectives

- Frame the problem as a binary classification task
- Explore and clean the dataset
- Apply feature engineering and selection
- Train multiple machine learning models:
  - Logistic Regression
  - Support Vector Machine
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Evaluate model performance using classification metrics
- Visualize model performance

## Data Description

The dataset contains survey responses from tech industry professionals regarding mental health conditions and treatment-seeking behavior. Key features include:

- Demographic information (age, gender)
- Employment characteristics (company size, self-employment status)
- Mental health history (family history, previous diagnosis)
- Workplace factors (mental health benefits, support for mental health)
- Work interference due to mental health conditions

The target variable is binary: whether the respondent sought treatment for a mental health condition.

## Installation and Usage
1. Clone this repository
2. Install the required packages:
  ```
  pip install -r requirements.txt
  ```
3. Run `main.py`

## Future Work

Several avenues for future development include:
- Time-based analysis to track how treatment-seeking behavior evolves
- Adding features like healthcare access and insurance coverage
- Developing a user-friendly interface for healthcare providers
- Applying explainable AI techniques for better model interpretation
- Transfer learning to leverage larger mental health datasets

## Limitations

- Sample limited to tech industry professionals
- Self-reported data subject to biases
- Cross-sectional data providing only a snapshot
- Trade-off between model performance and interpretability