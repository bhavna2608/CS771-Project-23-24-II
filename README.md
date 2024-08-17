# CS771-Project-23-24-II

In this project, we were given a XOR-PUFs with 3 PUFs but each PUF had 8 multiplexers i.e. the challenges were 8-bits long. Our task was to create a new feature vector from the 8-bit challenge so that a linear model
on that feature vector could predict the response. 

![image](https://github.com/user-attachments/assets/2b938433-4cdc-46e4-b0ad-a7ba1e178240)


First, using mathematical calculations, we proved that the 3 PUFs when combined could indeed fit a linear model. Then we applied two different linear models (Linear SVC and Logistic Regression) and did hyperparameter tuning to get the best possible accuracy and least amount of time to train the model. The respective hyperparameters we manipulated were as follows:

## 1. Linear SVC: 
i. Loss Function <br/>
ii. Regularization Parameter (c) <br/>
iii. Penalty Term <br/>
iv. Tolerance Criteria <br/>
v. Maximum Number of Iterations <br/>

## 2. Logistic Regression:
i. Inverse of Regularization Strength (C) <br/>
ii. Solver <br/>
iii. Penalty Term <br/>
iv. Tolerance Crieteria <br/>
v. Maximum Number of Iterations <br/>

## Conclusion: 
Based on our analysis, we come to the conclusion that LogisticRegression gives a better accuracy and takes lesser time in order to train the model when compared to LinearSVC.
