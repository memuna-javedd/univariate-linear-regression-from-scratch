# Univariate Linear Regression From Scratch

Predicting salary from years of experience with linear regression written in NumPy, with the cost function, partial derivatives and gradient descent all coded by hand.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/memuna-javedd/univariate-linear-regression-from-scratch/blob/main/Experience%26SalaryRegressionModel.ipynb)

## What it does
- Takes years of experience as the single input and predicts salary with the model `f(x) = w·x + b`
- Cost function (mean squared error), partial derivatives for `w` and `b`, and gradient descent written manually with loops, no scikit-learn
- Plots the data and the fitted regression line
- Notebook: `Experience&SalaryRegressionModel.ipynb`

## Dataset
`Salary_dataset.csv` (included in this repo): 30 rows with `YearsExperience` and `Salary`.

## Results
Trained with learning rate `0.01` for `10,000` iterations, starting from `w = 0` and `b = 0`.

| Parameter | Value |
|---|---|
| w | 9449.96 |
| b | 24848.20 |

- Predicted salary for 5 years of experience: about 72,098
- R² on the training data: about 0.957
- The learned values match the closed-form least-squares solution, so gradient descent converged

## Run it
Open the notebook in Google Colab using the badge above. To run it locally, install `numpy`, `pandas` and `matplotlib`, then open it with Jupyter.

## Status
Baseline model. It works end to end, and I am improving it as I learn.

## Planned improvements
- Add a train/test split, since the current R² is measured on the same data the model was trained on
- Track the cost at each iteration and plot the cost curve
- Replace the loops with vectorized NumPy
- Compare against scikit-learn's LinearRegression
