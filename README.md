# drug-rating-machine-learning
Machine learning analysis of drug ratings using Drugs.com data (Linear Regression, Random Forest, Logistic Regression).
# Drug Rating Machine Learning Analysis

This project analyzes drivers of drug ratings using user-reported data from Drugs.com.

## Objective
Identify structural factors that influence drug satisfaction and predict the probability that a drug receives a high rating (>=8).

## Dataset
Source: Drugs.com user review dataset  
Observations: ~200k drug reviews  
Features include:

- Drug class
- Pregnancy safety category
- Rx vs OTC status
- Number of side effects
- Activity level

## Methods

Three models were implemented:

1. Linear Regression – baseline interpretable model
2. Random Forest – nonlinear machine learning model
3. Logistic Regression – probability of high rating prediction

## Evaluation

Model performance was evaluated using:

- RMSE
- ROC Curve
- AUC
- Calibration Plot

## Key Findings

- Drug class is the strongest driver of user ratings
- Side effects have nonlinear impact on satisfaction
- Random Forest achieved the best predictive performance

## Tools

R  
Random Forest  
Caret  
ggplot2
