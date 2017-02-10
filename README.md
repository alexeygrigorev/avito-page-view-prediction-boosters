# Page View Prediction

## Avito BI Contest task 3

Solution for Avito Page View prediction competition (Avito BI contest task 3 on boosters)

- Competition website: https://boosters.pro/champ_4 (in Russian)


Solution:

- Linear SVM for all title tokens + one hot encoded params 
- LibFFM for the same features + date (LibFFM fork with regression: https://github.com/bobye/libffm-regression/tree/master/alpha-regression)
- Then stack these two models with weaker ones:
    - SVM on titles, SVM on params
    - ET on SVD and NMF
    - XGB as stacker
