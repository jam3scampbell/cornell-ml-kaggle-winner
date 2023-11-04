This was my winning submission (first place out of 155) to Cornell's ML Kaggle competition (https://www.kaggle.com/competitions/cs-4780-covid-case-hunters/leaderboard). The challenge was to classify a country's number of COVID cases based on demographic information in the low-data regime.

The intuition behind my solution is that we can estimate the predictive value of different data points using Kernel Ridge Regression, but then use gradient-boosted regression trees on those points with the most predictive value.
