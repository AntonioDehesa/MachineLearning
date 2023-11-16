# Week 1: Introduction

## Supervised learning
The training data you feed to the algorithm includes the desired solutions. Typically used for classification, or regression.
Examples: 
* k nearest neighbors
* linear regression
* logistic regression
* support vector machines (SVMs)
* Decision Trees and Random Forests
* Neural Networks

## Unsupervised learning
Unlabeled training data.
Used for clustering and anomaly detection, with algorithms such as kmeans, DBSCAN, Hierarchical cluster analysis, one-clas SVM, Isolation forest, PCA, Kernel PCA, Locally-linear embedding, t-distributed stochastic neighbor embedding, apriori, eclat. 

## Semisupervised learning
partially labeled training data. 


# Week 2: Data Analytics Project and process management

* KDD (Knowledge Discovery for Databases)
* CRISP-DM (CRoss Industry Standard Process for Data Mining)

## Defining the problem

* what is the business problem you are trying to address?
Translate the business problem into a technical problem. It is almost never perfectly defined, so ask questions

* how is it currently done?
If there is a method to do it, that is the benchmark. Your objective is to do better. Otherwise, just use that method. 

* what performance measure will you use to evaluate your model?
* what performance criteria do you need to meet to be considered successful?
Dont let perfect be the enemy of good enough. 
100% accuracy is perfect, but 99.99% is usually more than good enough. 

Frame it in terms of an analytical solution
* what kind of machine learning tasks is it?
* is it batch or online learning?

## Gather the data and explore the data

* Identify data sources and owners.
* Perform descriptive statistics
* Correlation analysis
Measures the strength and direction of the relationship between two variables. 
Correlated features may not harm performance, but they wont add extra information to the model, and will increase complexity.

* Visualizations, such as histograms. 


## Prepare data

* Handle missing data: Imputation, which can be done with the mean or median replacemente, or a decision tree using a feature with missing values as the target
Or removal of the missing data. 

* Transformation, such as one hot encoding
Skewness: Reduces the models predictability of the model to describe common cases because of textreme values in the tail. Does not matter with decision trees.

* Scaling
Always fit to training data and use fitted scaler to transform validation and testing data. 

* Data splitting

## Select the model

Use a champion-challenger approach
metrics to employ in decision
models need to perform well on test data, and future data. 
Sometimes you have a trade-off between explainability and performance. 

You cannot get rid of overfitting, you can only reduce it. 

## Tune the model

different models have settings called hyperparameters that you use to tune the architecture of the models. 
You can use grid searches to test multiple parameters of a model and check which combination of parameters is the best for that model. 
Then compare the best result of that model with the best results for other models.
Then you have the best model with the best possible parameters. 

You can also list and order the features in order of importance, and get rid of features that are not important. 

# Week 3: Classification

It is a supervised machine learning task.
Tries to predict one or more discrete classes as output
It is trained on input features and a known label

One of the most common ways to solve classification is with logistic regression.
Although called regression, it is a linear model for classification. Also known as logit regression, maximum-entropy classification, and log-linear classifier.


When training a model, accuracy is not a good measure when the data is highly skewed.
When it is skewed, you can categorize predictions as compared to actual labels. 
A common way to do this is with a confusion matrix. 

## Confusion Matrix

Common way to evaluate the performance of a classifier. 
Each row in a confusion matrix represents an actual class. 
Each column represents a predicted class.

## Performance measures

TP = True positives
TN = True negatives
FP = False positives
FN = False negatives

Accuracy: Ratio of correct predictions
 = (TP + TN ) / (TN + FP + FN + TP)

Precision: Accuracy when the model thinks it has it right
Use it when data is highly skewed or you want to minimizee false positives

= TP / (TP + FP)

Recall: Called sensitivity or true positive rate

= TP / (TP + FN)

Use to minimize the impact of false negatives. 

F1 Score

Harmonic mean of precision and recall

2 * (precision x recall) / (precision + recall)

It gives weight to low values
A high F1 score requires precision and recall to both be high

## ROC

Receiver Operating Characteristic

Evaluation tool used for binary classifiers
It plots true positive rate (recall) against the false positive rate
The False positive rate is the ratio of instances labelled negative that the model thought were positive
You want the curve to be as high as possible near the top left corner so you minimize the false positive even when recall goes up


# Week 4: Model Training

## Linear Regression Prediction

Equation: y_hat = theta_0 * x_0 + theta_1*x_1... + theta_n*x_n = theta.x
where 
y_hat = predicted value
n = number of features
theta_j = j^th model parameter
theta_0 = bias term


### Minimizing the Cost Fuction for Linear Regression

training a model tweaks the parameters to get the best fit on the dataset
you need a performance measure to determine how good the model is
a common performanace measure is the mean squared error (MSE)
We need to reduce this error

How to reduce it: 
* Normal Equation: If there are more examples than features, there is a direct vector solution to solving for the minimized parameters directly
Theta_hat = (transpose(x)*x)^-1 * transpose(x)y


* Calculation of pseudoinverse
singular value decomposition
* Gradient Descent


## Gradient Descent

* tweaks parameters iteratively to minimize a cost function
* finds which direction is down and takes a step in that direction
* the gradient will be 0 at the bottom

### Learning rate

start by filling theta with random values
minimize the cost function gradually
how quickly you go down is determined by the learning rate
not too big or small

problems with:
* local minimums
* plateaus
* MSE is convex, which is good
may require features to be scaled

### Types of gradien descent

#### Batch gradient descent

* based on partial derivatives
* it changes theta a little in each step for the entire data
* to go down, you go the opposite of the gradient vector (substract it from theta) multiplied by the learning rate
* slow, but faster than calculating the normal equation

#### Stochastic Gradient Descent

it picks a random example in the training set and computes gradients on that one instance
faster, but it may bounce up and down as it decreases on average
this bouncing helps getting out of local minima, but it may jump out of the global minima
this can be solved by reducing the learning rate the more steps it takes

#### Mini batch gradient descent

middle ground between batch and stochastic
the gradients are computed on small random sets of instances called mini-batches

## Polynomial Regression

uses a linear model to fit nonlinear model by adding powers of each feature as new features in the training set
you train the model on the extended set.

## Learning Curves

plots a model performance on training and validation sets as a function of training set size
the model is trained several times on different sized subsets of data
the error increases as the size of the subset grows until it plateaus because it cant be solved with a linear equation

a learning curve where both the training and evaluation curves plateau but are close and fairly high is a sign of underfitting
overfitting on the training set causes its curve to be much lower than the error on the validation data

## Regularized Linear Models

it seeks to limit the size of the weights of the model
there are 3 types of regularization for linear models:
* ridge regression
* lasso
* elastic net

you can also use early stopping

### Ridge

it adds a new tero to the loss function
the term squares the sum of the parameter vector and scales the amount of shrinkage with a new hyperparameter called alpha
the regularization term is only added to the loss function during training
also called L2, because we are squaring the vector
the data should be scaled
increasing alpha leads to flatter predictions. it also reduces variance but increases bias

### Lasso
it uses the absolut value of the L1 norm of the weight vector as a regularization term in the cost function

only keeps important features

# Week 5: Support Vector Machines

Linear and non-linear classification
regression
outlier detection
uses kernels to transform features
good for classification of complex small to medium sized datasets

identifies boundaries between linearly separable classes


## Benefits

* good for data with lots of features
* remains solvable even when number of features is greater than number of samples
* memory efficient since it only uses a subset of data called support vectors in the decision function
* different kernels can be used to define decision function, and you can even specify custom ones

## Disadvantages

* Sensitive to outliers
* features must be scaled
* overfitting can happen. specially is you have more features than samples
* they do not provide probability estimates

## Linear SVM Classification

2 approaches to how we treat data on or near the road (the space that divides the categories we are classifing)
* hard margin classification: does not allow any points on the road. insists different classes are on different sides of the road (no overlap)
* soft margin classification: it solves for the widest street possible, but the limits are flexible, such as objects on the road, or even the wrong classification

### Hard margin issues

it assumes that the data has to be linearly separable. if it is not, it has issues
outliers near the road can make the margins of the hard margin solution small
outliers near the other class may make it unsolvable

## Non-linear SVM Classification


## SVM Regression

we try to fit as many instances as possible on the street. 
margin violations are defined as points off the street

## SVM Training

The slope of the decision function is equal to the norm of the weight vector
if we lower slope the points where the decision function is equal to +-1 are going to be twice as far from the decision boundary
the smaller the weight vector w, the larger the margin
so if we want a large margin we have to minimize the norm of the weight vector

### Hard margin objective

we want to minimize the square of the weight vector norm
add a constraing to force the points to be greater than one or less than minus one

### Soft margin objective

measures the distance of the point on the street to its marginal hyperplane
we want to minimize the square of the weight vector
add the sum of the slack variables and multiply that by C
a high C will foce the slack variables to be smaller, allowing less margin violations


# Week 6: Decision Trees

* intuitive
* easy to interpret
* can perform both classification and regression
* handles both numerical and categorical data
* can handle multi-output problems

## How do Decision tree regressors work

* Data is sorted by feature
* Calculate mean as cut point for the first feature
* split the data using the cut point and move to the next features
* do the same thing

## Predictions

you start at the root node, then the input data follows the rules and instructions for each node until it reaches the leaf with the closest predicted class / output

## Entropy

type of impurity measure
it is zero when it contains instances of only one class
it is the negative sum of each of the class values/samples times the log_2 of the values/samples


## Gini index

It measures the probability for a random instance being misclassified when chosen randomly. 
The lower the index, the better the lower the likelihood of misclassification. 

Gini = 1 - sum from i = 1 to j of P(i)^2 
where j = number of classes in the target variable
P(i) = ratio of pass/total number of observations in node

## CART Training

it splits the training into two subsets using a single feature k and a threshold. 

Also called a growing tree
it starts at the base of the tree
it iterates at every branch until it reaches the maximum depth or it cannot get impurity to go down anymore

## Regularization

decision trees tend to overfit the training data
as the structure of the tree is not known before the tree is built it is said to be non-parametric
compare this to a linear model, which has limited degrees of freedom
the linear model is parametric
parametric models tend to cost less when fitting the data


# Week 7: Ensemble trees

## Ensemble Learning

Follows "wisdom of the crowd": asks 1000 nodes a question, averages the answers, and the average answer is expected to be better than the answer of a single node.
Example: 
* train a group of decision tree classifiers, each on different random subsets of the training data
* obtain the predictions of all the individual trees and predict the class that gets the most votes
* this is an example of random forest, since it uses an ensemble of decision trees

In real projects, you should use the ensemble model towards the end of a project, when you have some good predictors, combine them in an ensemble model.

## Voting Classifier

Machine learning model that trains on an ensemble of numerous models and predicts an output based on the predictions of the other models. 

### Voting methods
* hard: uses a majority of votes to decide the predicted class
* soft: uses probabilities to choose the predicted class

## Bagging and Pasting

### Bagging
Bagging = Bootstrap aggregating
random sampling with replacement. It allows an individual predictor to see a certain instance several times.
Seeing some instances more than once means that some instances will not be seen even once. 

The aggregation calculated for bagging in a classification problem is the mode
For regression, it is the average.
The result of limiting individual predictors to train on training subsets increases bias (number of errors) they experience
But individual predictors will also have less variance on the random subset of data

#### Out of Bag Evaluation

The training instances not seen by a predictor during bagging can be used to evaluate the performance of that predictor. This is called out of bag (oob) evaluation
Probability of an instance being selected: P(selected) = 1/m

### Pasting
Random sampling without replacement. 

## Random Forests

Tipically use bagging.
Individual trees have large variance, aer prone to overfit, and do not generalize well.
We want independent and diverse decision trees. 


## Boosting

Ensemble method that combines multiple weak learners into a strong learner
Instead of building entire trees as the base estimator, boosting uses stumps, which have one branching node, and two leaf nodes off of it with max depth of 1
These stumps are by definition weak learners
Predictors are trained sequentially
New predictors correct previous ones

### Boosting algorithms

* Adaboost: Pays attention to underfitted training instances
Improves previous predictors by looking at the instances it got wrong
Additional predictors added sequentially look for and find instances that are harder and harder to classify
Uses stumps
It also uses weights for each predictor
Predictors with high accuracy get a higher weight
At prediction time, adaboost computes the predictions of all the predictors, applies the weights to them, and adds the results to come up with a final aggregated prediction
* Gradient Boosting: Fits the new predictor to the residual errors of the previous predictor
Insted of updating instance weights, it builds a new predictor to the residual errors and targets of the initial one


To handle overfitting, you can reduce the number of predictors

There is a library called XGBoost, which uses an optimized version of gradien boosting

## Stacking

The aggregation of the predictions is handled by the model itself
The portion of the model that trains the aggregation is called a blender
the training set is broken into two subsets
the first layer is trained on the first subset of data
it then makes predictions on the second subset of training data

### Training the blender
a new training set is built using the predicted values as new input eatures in addition to the target features
the blender is trained on this new dataset, so it learns to predict the target value based on the first layers predictions

## Bagging versus Boosting

both ensembles
bagging builds many predictors simultaneously
boosting builds weak predictors sequentially to reduce the errors/misclassifications from the previous predictors

# Week 8: Clustering K-means

## Representative-Based Clustering

* Clustering: Process of grouping similar items together in representative-based clusters
* items are partitioned in k groups
* the dataset is described as n points in d-dimensional space
* the centroid is a point that represents the summary of the cluster
* the stirling numbers of the second kind produces the exact number of partitions of n points
* the points are organized into k nonempty and disjointed groups
* since any point can be assigned to any of the k clusters, it is possible to have O(k^n / k!) clusters
* this proves that brute-force is not a possible solution for clustering
* we overcome these issues with two approaches: K-means, and expectation-maximization algorithms

### Scoring Function
SSE(C) = sum from i = 1 to k of the sum of x_j E C_i of the squared absolute of X_k - u_i

### K-means

Steps: 
* cluster assignment
* centroid update
* each point is assigned to the closest mean
* each point is assigned to the cluster C_j, where j = arg_min from i = 1 to k of the squared absolute of X_j - u_i

## Kernel K-means

* allows for non-linear boundaries
* this technique detects nonconvex clusters
* the technique maps data points high-dimensional space using non-linear mapping
* the kernel trick allows for feature space to be explored by the function using dot product
* it can be operationally expensive
* O(n^2) to compute the average kernel value for all clusters
* O(n^2) to compute the average kernel cluster for each k
* total complexity is O(tn^2) where t = iterations