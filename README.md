# Survival-Prediction-of-Horse-Colic


## Introduction

Colic in horses, simply refering to abdominal pain, is one of the most common problems in equine practice. It has a significant economic impact on the racehorse industry and is a major concern for owners.

Equine colic can be divided into 2 major categories; gastrointestinal and nongastrointestinal. Horse’s vital signs (heart rate, respiratory rate, and mucus membrane color) is important information for diagnosis. Nongastrointestinal colic cases can usually be excluded based on physical examination findings; these include signs of abdominal discomfort due to urinary urolithiasis and disorders of reproductive, nervous, respiratory, or musculoskeletal systems. Causes of gastrointestinal colic (GC) are gut distension, tension on the root of mesentery, ischemia, deep ulcers in the stomach or bowel, and peritoneal pain.

The purpose of this study was to explore the symptoms of horses with colic and predict survival, with a view to providing equine practitioners with a better reference to equine colic condition, which may contribute to timely treatment and higher colic survival rates.

## Data Description

The pathology data of 368 horses presented with signs of colic is reviewed. It includes 28 attributes about the vital signs and surgical information of horses including rectal temperature, mucous membranes, nasogastric reflux, etc. The attributes can be continuous, discrete, and nominal. 

* surgery or not
* Age
* Hospital Number
* rectal temperature
* pulse, the heart rate in beats per minute
* respiratory rate
* temperature of extremities
* peripheral pulse
* mucous membranes
* capillary refill time
* pain - a subjective judgement of the horse's pain level
* peristalsis
* abdominal distension
* nasogastric tube
* nasogastric reflux
* nasogastric reflux PH
* rectal examination - feces
* abdomen
* packed cell volume
* total protein
* abdominocentesis appearance
* abdomcentesis total protein
* outcome
* surgical lesion
* type of lesion
* cp data

## Data Preparation

We combine the training data and the testing data together, so the data preparation process only needs to be done once and all the statistics are more accurate.

The response variable is the outcome of the horses, a categorical variable showing whether the horse lived or died or was euthanized.

### Data Cleaning

There are 30\% of missing values. For data cleaning process, we did several processes:

* Delete two observations without the response variable "outcome".
* Delete three features with more than 50\% missing values: nasogastric reflux PH, abdominocentesis appearance and abdomcentesis total protein.
* Delete two explanatory variables irrelevant to the prediction: hospital number and cp data.
* Replace missing values left with 0. The reason for it is that for logistic regression, predictors being zero will not affect the prediction result because $Sigmoid(0)=0.5$.

### Data Conversion

Then we convert the response variable "outcome" to a binary variable. Only the alive horses will be 1 and those died or was euthanized will be 0.

For ordinal variables, we treat them like continuous variables because we don't want to lose information in the ordering. For example, temperature of extremities, a subjective indication of peripheral circulation, is an ordinal variable with possible values: 1 = Warm, 2 = Normal, 3 = Cool, 4 = Cold. We assume the numerical distance between each temperature categories is equal when treating the variable as continuous instead of discrete. Even though the assumption may not be close to reality, it's the order of the temperature that associates with causes of colic because cool to cold extremities indicate possible shock while hot extremities should correlate with an elevated rectal temp, which may occur due to infection.

Also, we have to convert the categorical variables into numeric using one hot encoding because XGBoost can only deal with numerical variables. We created dummy variables for categorical variables such as surgery history, age (young or adult), mucous membranes and so forth, and substitute the categorical variables with their corresponding sparse matrices.

After all the data manipulation, we split the data into two sets: the training set with the first 299 observations and the testing set with the 67 observations left.

## Logistic Regression

We use logistic regression as a benchmark model. For logistic regression, we didn't do any feature selection in order to get the best prediction performance. The training accuracy is 0.7659 and the testing accuracy is 0.7015. Logistic regression tends to predict more alive cases than there truly are.

## XGBoost

### Introduction of XGBoost

XGBoost, short for eXtreme Gradient Boosting, is a popular implementation of Gradient Boosting because of its speed and performance.

For classification problems in this case, we use booster = gbtree parameter, that is, a tree is grown one after other and attempts to reduce misclassification rate in subsequent iterations by giving the next tree a higher weight to misclassified points by the previous tree. In this case, we use binary classification error rate as the evaluation metric.

Every parameter is significant to the performance of the XGBoost model. Most frequently used and tunable parameters used in this paper are explained as belo:

* **nrounds**: It controls the maximum number of iterations. For classification, it is similar to the number of trees to grow. Should be tuned using CV.
* **eta**: It controls the learning rate, i.e., the rate at which our model learns patterns in data. After every round, it shrinks the feature weights to reach the best optimum. Lower eta leads to slower computation. It must be supported by increase in nrounds. Typically, it lies between 0.01 - 0.3.
* **max\_depth**: It controls the depth of the tree. Larger the depth, more complex the model; higher chances of overfitting. There is no standard value for max\_depth. Larger data sets require deep trees to learn the rules from data.
 * **min\_child\_weight**: In classification, if the leaf node has a minimum sum of instance weight (calculated by second order partial derivative) lower than min\_child\_weight, the tree splitting stops. In simple words, it blocks the potential feature interactions to prevent overfitting.
* **subsample**: It controls the number of samples (observations) supplied to a tree. Typically, its values lie between (0.5-0.8).
* **colsample\_bytree**: It control the number of features (variables) supplied to a tree. Typically, its values lie between (0.5,0.9).

### Initial Model by Default

Before any hyper tuning, we fit the model on the training set using default parameters. Except for number of rounds, which we use cross validation to tune as 10. The training error for this model is 0.0201 and the validation error is 0.2537. And the testing error is 0.3134. Therefore if XGBoost model is not tuned well, it can be less accurate than logistic regression.

### Tuning Parameters
Because the validation error and testing error are high while the training error is only 0, there is severe over fitting problem with this model. There are in general two ways to control overfitting in XGBoost:

* Control model complexity directly, including max\_depth, min\_child\_weight and gamma.
* Add randomness to make training robust to noise, including subsample and colsample\_bytree.


We can also reduce stepsize eta, the learning rate, and increase num\_round for this.

Grid search and manual search are the most widely used strategies for hyper-parameter optimization. Random search differs from grid search mainly in that it searches the specified subset of hyperparameters randomly instead of exhaustively. The major benefit is comparably decreased processing time. There is a tradeoff to decreased processing time that we aren’t guaranteed to find the optimal combination of hyperparameters. However, empirically and theoretically, randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid. XGBoost is famous for it's computational efficiency, so instead of grid search, we use MLR package in R to perform random search to find the best parameters. In random search, we'll build 10 models with different parameters, and choose the one with the least error.

### Final Model and Results
We set eta as 0.1 along with nrounds equals to 100, the maximum depth of the tree as 5, minimum child weight as 3.63, the number of observations supplied to a tree as 0.5813 and the number of variables supplied to a tree as 0.6754. Then the tuned model is less over-fitting than the initial one. The training error is 0.0970 but the testing error is just 0.1940, significantly more accurate.

The importances of the top 15 most important variables are not exactly the same as the model with default parameters. The tuning process changes a little of the weight we put on different parameters and their relative importance remains nearly the same. It makes sense in reality that the first lesion is the most important and most frequent variable to determine whether the horse is going to survive or not, and there are nearly one fifths of the observations are classified based on this feature. Then comes the variables about the vital signs of horses including total protein, the pulse, respiratory rate and the rectal temperature, etc.

The break down plot explains how each input feature contributes to the prediction. Beginning and end of each rectangle corresponds to the prediction with and without particular feature. From the break down plot, we can see exactly how each feature has influenced the prediction. The intercept equals to 0.592, so the probability of horse survived is higher, which may due to the unbalanced data. If the horse has inflammation all over its intestinal sites, that is, the variable "lesion1" equals to 11300, then the probability of survival will decrease by 0.092. If the temperature of extremities is cool or cold, then the probability will decrease by 0.083, because cool to cold extremities indicate possible shock. Similarly, if the horse has 45 or more red cells by volume in the blood, then the probability of survival will increase by 0.055. And if the rectal temperature of the horse is higher than 38.5, the probability will increase by 0.082, because temperature may be reduced when the animal is in late shock. We can also conclude from the break down plot that the age of a horse with colic does not affect its survival rate.


However, the total accuracy of XGBoost model is only 0.806, which may due to the lack of data and the large amount of missing data. XGBoost model does not obviously predict the horse to be alive as logistic regression does.

## Discussion of Results

L. Curtis, I. Trewin, G. C. W. England, et al did a survey about veterinary practitioners’ selection of diagnostic tests for the primary evaluation of colic in the horse. 248 responses were collected from UK equine practitioners. Their results show that participants used response to analgesia/treatment most frequently in colic cases, as about 87.2 percent of cases, followed by rectal examination for about 75.9 percent of cases, nasogastric intubation for 43.9 percent of cases, haematology and biochemistry for 15.2 percent and abdominal paracentesis for 13.5 percent.

The most commonly identified reason practitioners would use rectal examination was to identify lesion or case type. Abdominal paracentesis was considered most useful to differentiate ‘medical versus surgical’ or to determine prognosis, nasogastric intubation most useful for suspected ‘proximal’ intestinal lesions, ultrasound was also considered most useful for identifying lesion or case type, haematology/biochemistry most useful for recurrent/chronic cases and response to analgesia was considered most useful in most cases.

However, some of the popular diagnostic tests do not play an important role for prediction in our model.. The coverage of some diagnostic tests mostly used by practitioners is really low, which means that relatively not much of observations are classified based on this test.

Also, when vets make diagnosis based on nearly all the relevant features instead of just some of then like in the model. Models that considering all the inputs may be better for interpretation.

Therefore, even though the model output is useful for decision making at some degree of probability, it just provides some references rather than the answer.

