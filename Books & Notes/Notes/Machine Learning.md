# Machine Learning

### What is Machine Learning
Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without human intervention or assistance. Machine learning focuses on the development of computer programs that can access data and look for patterns in data and make better decisions in the future based on the examples that we provide.

Machine learning algorithms are learning on the training set to reduce the training error. Trained models is to produce the test error on the test set, which is the expected value of the error on the new input.

Hyperparameters: the parameters we choose that can use to control the behavior of the learning algorithm. Instead of learning it from data, we select it using the subset of data, validation set.

### Unsupervised vs. Supervised
Unsupervised Learning: given a dataset containing many features, learn useful properties of the structure of this data set.

Example:
* Learning to draw examples from a distribution
* learning to denoise data from some distribution
* finding a manifold that the data lies near
* clustering the data into groups of related examples

Supervised Learning: given a dataset containing features and responses/labels, train a model by learning each data points with responses.

## Data Processing
### Handling Outliers
**Outliers**: extreme values in target variables
**Influential**: extreme values in independent variables that do have high leverage to model performance
* Exploratory
    1. Multivariate analysis: scatter plots
    2. Box-Plot: useful low dimensional data
    3. Cook’s Distance: measures the effect of deleting a given observation for finding influential points.
    4. Z-Score: normally define outliers whose z-score is greater than a threshold value.
* Solution
    1. Winsorizing: replace the extreme values of an attribute to some specified value. For exmaple, replacing two tail 5% data by a value. 
    2. Transformation: convert to normal distribution to reduce the variability of outlying observation, and also improve normality-base models, but causes large error in prediction.
    3. Tree-based, SVM model are robust to outliers
    4. Robust metrics: MAE, truncated loss

### Handling Missing Data
1. Imputation using mean, most_frequent, median and constant for numeric variables, or most frequent value for categorical variables: mean might cause overfitting
2. Imputation using a randomly selected value
3. Imputation with a model (The k nearest neighbor algorithm is often used to impute a missing value, but the euclidian distance is time consuming.)
4. Removing large proportion missing data: lose valuable information, so the consequence is overfitting in the training dataset and worse prediction performance on the test dataset.

### Imbalanced Data
1. Over-sampling: It refers to generating data for smaller class by sampling with replacement. (Tip: Duplicating Rows may help)
2. Under-sampling: It refers to decreasing the number of rows for larger size class to make training more balanced
3. Generating data using SMOTE : Synthetic Minority Over-sampling Technique works on generating new data points for smaller class
4. Class Weights: Some algorithm allows you to assigns different weights to class to improve training

### Data Transformation
Depending on the model, skewness may violate model assumptions (linear regression, logistic regression) or may impair the interpretation of feature importance and inference, so the Normality testing is helful.
1. Shapiro-Wilks test: scipy.stat.shapiro
2. Fisher–Pearson standardized moment coefficient: .stew()
3. Common Transformation: (Square root, Reciprocal, Log, Box Cox)

### Category Encoding
1. One-hot Encoding: the categorical variable is broken into as many features as the unique number of categories for that feature and for every row, a 1 is assigned for the feature representing that row’s category and rest of the features are marked 0
    * Curse of Dimensionality when there are many categories, causing model overfiting
    * Sparse Representation: efficient, but require models support sparse matrics (xgboost, LGBM)
    * Sparsity loses features power, continuous variables usually have higher feature importance
    * Dense Representation: require heavy memory
2. Numeric (label) encoding: simply converting each category in a column to a unique number 
    * Ordinal data can be rank ordered
3. Binary encoding: it's a combination of numberic encoding and one hot encoding to covert a binary code
    * Create fewer features than one-hot, while preserving some uniqueness of values in the the column
    * Work well with higher dimensional ordinal data
    * Sparsity loses features power, continuous variables usually have higher feature importance
4.  Deep Encoding: map each of the unique category into a N-dimensional vector real numbers
    * embedding_size = min(50, m+1/ 2)
    * Able to control the number of dimensions to represent the categorical feature

## Performance Metrics
### MAE (Mean Absolute Error)
MAE measures the average magnitude of the errors in a set of predictions, without considering their direction.
$$
MAE = \frac{1}{n}\sum|\hat{y}-y|
$$
* Same unit of measurement as the response variable (easy interpretable)
* Equal weight for all errors
* No direction

### R-sqaured (coefficient of determination)
The proportion of the response variable variation explained by the model.
* R-squared = Explained variation / Total variation = $1 - \frac{RSS}{TSS}$
* R-squared = $\frac{Cov(X, Y)}{var(Y)}$
* $R^2 \subset (-\infty, 1]$, negative ocurrs when RSS > TSS that prediction is worse than $\bar{y}$
* Always improves $R^2$ if add independent variables
* Very sensitive to outliers

### RMSE (Root Mean Square Error)
RMSE is a measure of how data is closed to the best fitted line. It is the standard deviation of prediction error, and direct correlation with the variance of frequency distribution of errors. RMSE has more strict behavior than MAE with large errors, namely it tells us RMSE is sensitive to error distribution of samples.

$$
MSE = E[(\hat{\theta} - \theta)^2] = Var(\hat{\theta}) + Bias(\hat{\theta})^2
$$
* Same unit of measurement as the response variable (easy interpretable)
* High weight to large errors: more useful when large errors are particularly undesirable.
* No direction
* MAE ≤ RMSE ≤ MAE * sqrt(n), where n is the number of test samples.

### Cross Entropy
Measures the difference between two probability distributions, or the performance of a classification model whose output is a probability value between 0 and 1.
$$
H(p,q) = -\sum{p\ log\ p}
$$

* Skewed Probability Distribution: Low entropy.
* Balanced Probability Distribution: High entropy.

### Gini Index
Measures how often a randomly chosen element from the set would be incorrectly labeled
$$
Gini = 1 - \sum{p^2}
$$

* Gini's max impurity = 0.5
* Gini's max purity = 0

### Accuracy
It measures how many observations, both positive and negative, were correctly classified. It shouldn’t be used on imbalanced problems because it's easy to get a high accuracy score.
* Balanced problem
* Every classe is equally important

### Precision (Specifility)
The fraction of positive prediction by the model that were correct (more hits of positives). What is the confidence/reliability of the model?
$$
\frac{TP}{TP+NP}
$$

### Recall (Sensitivity)
The fraction of true events that were detected (more hits of reality). What is the ability of the model to have correct predictions?
$$
\frac{TP}{TP+NF}
$$

### ROC
The plot is consisted of TP vs. FP, provide a principled mechanism to explore operating point tradeoffs.

### F Score
It combines precision and recall into one metric by calculating the harmonic mean between those two. It is actually a special case of the more general function F.

$F_\beta = \frac{(1+\beta^2)*precision * recall}{\beta^2*precision + recall}$

* Higher $\beta$ choose, the more care about recall (false-negative) over precision
* Every binary classification problem where you care more about the positive class
* Can be easily explained to business stakeholders

### ROC AUC
It is a chart that visualizes the trade-off between true positive rate (TPR) and false positive rate (FPR). Basically, for every threshold, we calculate TPR and FPR and plot it on one chart. The higher TPR and the lower FPR is for each threshold the better and so classifiers that have curves that are more top-left-side are better. In order to get one number that tells us how good our curve is, we can calculate the Area Under the ROC Curve, or ROC AUC score. The more top-left your curve is the higher the area and hence higher ROC AUC score.

Alternatively, **ROC AUC is especially good at ranking predictions.** The ROC AUC score is equivalent to calculating the rank correlation between predictions and targets. From an interpretation standpoint, it is more useful because it tells us that this metric shows how good at ranking predictions your model is. It tells you what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance.
* Should use when care equally about positive and negative classes. If care about true negatives as much as we care about true positives then it totally makes sense to use ROC AUC.
* Should use it when ultimately care about ranking predictions.
* Should not use when data is heavily imbalanced. The intuition is the following: false positive rate for highly imbalanced datasets is pulled down due to a large number of true negatives.

### PR AUC (Average Precision)
It is a curve that combines precision (PPV) and Recall (TPR) in a single visualization. For every threshold, you calculate PPV and TPR and plot it. Similarly to ROC AUC score, you can calculate the Area Under the Precision-Recall Curve to get one number that describes model performance. The higher on y-axis your curve is the better your model performance.

The PR AUC is also considered as the average of precision scores calculated for each recall threshold, so also to adjust the business decisions by choosing/clipping recall thresholds.
* Heavily imbalanced: PR AUC focuses more about positive than negative class.
* To communicate precision/recall decision to other stakeholders.
* To choose the threshold that fits the business problem.

### Bias & Variance Trade-off
Underfitting: occurs when the model is not able to obtain much information from features, so results in high error value on the training set, which is high bias low variance

Overfitting: occurs when the model complexity increases, results in increasing variance and decreasing bias, so the gap between training and test erorr is too large.

Try regularized model
Try tuning the hyper-parameters
Try Simpler models
Inducing noise in training data can help sometimes
Try getting more training data
Using Early stopping if algorithms allows
Using Cross Valiadation
Pruning in tree based

### Cross Validation
Cross validation is a resampling procedure used to evaluate the performance of models on unseen data in order to estimate how the model is expected to perform in general, producing more generalized models and avioding overfiting issue.

1. Randomly splits data into k equal sized subsamples
2. Train the model on k-1 subsamples and predict the hold-out kth subsamples
3. Repeated k times, holding out each subsample once as the validation set
4. Evaluate performance by averaging k results to produce a single estimation.

### Regularization
Any modification we make for machine learning algorithms reduce test error but not training error.

**L1 Regualization (Lasso)**: Least Absolute Shrinkage and Selection Operator.
1. Shrinks the less important feature’s coefficient to zero, producing sparse solutions
2. Performs feature selection in case a large set of features.
3. Gradient of Lasso $\frac{dL_1}{dw} = \pm{w}$, which is no longer scales linearly with w, instead a fixed weight decay. It moves any weights towards 0 with the same step size, regardless the weight's value.
4. Slow converge

**L2 Regularization (Ridge)**: Adds “squared magnitude of the coefficient" as penalty term to the loss function. The regularization parameter
$\lambda$ penalizes all the parameters so that the model generalizes the data and overcome overfit.

1. $\lambda$ is very large, then lead to underfitting
2. Forces the weights to be small but does not make them zero and does not give the sparse solution
3. Gradient of Ridge $\frac{dL_2}{dw} = w$, which is linearly decay. It moves any weight towards 0, but it will take smaller and smaller steps as a weight approaches 0.
4. Solves multicollinearity issue
5. Not robust to outliers as square terms blow up the error differences of the outliers
6. Performs better when all the input features influence the output, and all with weights are of roughly equal size

## Unsuprivised Models

### Principal Component Analysis (PCA)
An orthogonal linear transformation, "principal components" that transfers the data to a new coordinate system. It can be used to reduce dimensions of data without much loss of information. PCA provides a list of the principal components that keeps the largest variance by any projection of data to describe how the original data is distributed.

1. Centerize the data
2. Calculate covariance matrix $C = X^TX$ 
3. Compute the unit eigenvectors $V$ of $C$ by SVD
4. Sort eigenvalues and select top K principle components
5. Transform data to new system: $XV$

Standardizing is very important for PCA because of the way that the principle components are calculated by Singular Value Decomposition, which finds linear subspaces which best represent your data in the squared sense. By centering our data, we guarantee that they exist near the origin, and it may be possible to approximate them with a low dimension linear subspace. 

### K-Means
The k-means algorithm searches similar items and groups them into K clusters for an unlabeled dataset. The idea of the K-Means algorithm is to find K-centroid points and every point in the dataset will belong to the clusters having minimum distance.

It accomplishes this using a simple conception of what the optimal clustering looks like:
* The "cluster center" is the arithmetic mean of all the points belonging to the cluster.
* Each point is closer to its own cluster center than to other cluster centers.

The Elbow method is a technique to select the optimal number of clusters by fitting the model with a range of clusters.
* Distortion: sum of squared distances from each point to tis assigned center
* Silhouette: ratio of intra-cluster and nearest-cluster distance
* Calinski_harabasz: ratio of within to between cluster dispersion

Silhouette Score
It's used to evaluate the quality of clusters such as K-Means in terms of measuring how well a point is smiliar to its own cluster compared to other clusters. To calculate the Silhouette score ($S$) for each observation, we need two distance metrics:
* Mean intra-cluster distance ($a$): average distance between the observation and all other data points in the same cluster
* Mean nearest-cluster distance ($b$): average distance between the observation and all other data points of the next nearest cluster

$$
S_i = \frac{b_i - a_i}{max(a_i, b_i)}
$$
The value of the score varies from -1 to 1, and a high value indicates the cluster is dense and well-separated than other clusters. A value near 0 represents the observation is very close to the decision boundary of the neighboring clusters. A negative score means the observation has been assigned to the wrong clusters. 


### T-SNE
A machine learning algorithm for non-linear dimensionality reduction technique and exploring high-dimensional data visualization.

Properties:
1. Embedding high-dimensional data into a space of two or three dimensions, which can be visualized in a scatter plot.
2. Similar objects are modeled by nearby points and dissmiliar objects are modeled by distant points.
3. Hyperparameter perplexity 5-50 should be smaller than the number of data points.
4. Cluster sizes, distances between clusters are not meaningful.

Step:
1. Construct a probability distribution over pairs of high-dimensional objects that similar objects have a high probability of being picked
2. Define a similar probability distribution over the points in low-dimensional map, and minimize the divergence between two distributions with respect to the locations of points in the map.

## Suprised Models

### Linear Regression
A linear approach to modeling the relationship between a continous response and one or more explanatory variables (predictors).

Assumption:
1. linear relationship between predictors and target variable
2. Gausian noise: error are independent, constant variance
3. No multicollinearity

### Logistic Regression
A statistical model that use the logistic sigmoid function to model a binary label. It constructs a model which estimates probability of each class.

1. Sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}} \rightarrow \hat{p}(x)$ 

### Support Vector Machine (SVM)
Construct a optimal hyperplane such that maximize the distance (margin) to the cloest point in the each class.

**Kernel Trick**: Applying the kernel transformation (non-linear function) to all inputs, mapping all data into high-dimensional space, then learning a linear model in the new transformed space.

### Decision Tree
Tree-building algorithm is a subalgorithm that splits the samples into two bins by selecting a variable and a value. This splitting algorithm considers each of the features in turn, and for each feature selects the value of that feature that minimizes the impurity of the bins.

### Gradient Boost Decision Tree (GBDT)
A machine learning algorithm that iteratively constructs an ensemble of weak decision tree learners through boosting technique. The misclassified records are given greater weight (‘boosted’) to correctly predict them in later models. These weak learners are later combined to produce a single strong learner.

Each weak learner uses Gradient Descent optimization to update the weights in the model to reach a local miniment of the cost function.

Algorithm:
1. Initialize model with a constant value
2. Compute so-called pseudo-residuals
3. Fit a base learner to pseudo-residuals, i.e. train it using the training set
4. Compute multiplier $\gamma$ by solving the optimization problem of loss function
5. Update the model
6. Combine all weaker learning to a strong learner

### Multi-Class Classification
**One vs. One**:

**One vs. All**:

**Softmax**: Takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one, which represent normalized probability distribution over n different classes.

$P(y_i|x) = \frac{e^x}{\sum_i^n e^x}$







