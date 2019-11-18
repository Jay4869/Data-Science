# Machine Learning Concepts

### What is Machine Learning
Machine learning algorithms are learning on the training set to reduce the training error. Trained models is to produce the test error on the test set, which is the expected value of the error on the new input.

Assumptions:
* training and test data set are independent from each other
* training and test data set are identically distributed, drawn from the same shared probability distribution.

Goals:
* minimize the training error
* minimize the gap between training error and test error

Hyperparameters: the parameters we choose that can use to control the behavior of the learning algorithm. Instead of learning it from data, we select it using the subset of data, validation set.

### Unsupervised vs. Supervised
Unsupervised Learning: given a data set containing many features, learn useful properties of the structure of this data set.

Supervised Learning: given a data set contianing features and responses/labels, train a model by learning each data points with responses.

### Performance Measure
Accuracy: the proportion of examples for which themodelproduces the correct output.

Error rate: the proportion of examples for which themodel produces the incorrect output.

**R-sqaured (coefficient of determination)**: a statistical measure of how close the data are to the fitted regression line. Also, the percentage of the response variable variation that is explained by a linear model.
1. R-squared = Explained variation / Total variation = $1 - \frac{RSS}{TSS}$
2. R-squared = $\frac{Cov(X, Y)}{var(Y)}$
3. $R^2 \subset (-\infty, 1]$, negative ocurrs when prediction is worse than $\bar{y}$
4. Always improves $R^2$ if add independent variables

**MSE (Mean Square Error)**: how close a regression line is to a set of points. It's calculated by taking the squared distances from the points to the regression line.

$MSE = E[(\hat{\theta} - \theta)^2] = Var(\hat{\theta}) + Bias(\hat{\theta})^2$

**0-1 Loss**

**Cross Entropy**: Measures the performance of a classification model between a “true” distribution $p$ and an estimated distribution $q$, whose output is a probability value between 0 and 1.

$H(p,q) = -\sum_xp(x)\ log\ q(x)$


### Bias & Variance Trade-off
Underfitting: occurs when the model is not able to obtain much information from features, so results in high error value on the training set, which is high bias low variance

Overfitting: occurs when the model complexity increases, results in increasing variance and decreasing bias, so the gap between training and test erorr is too large.

### Cross Validation
The goal of cross-validation is to test the model's ability to predict new data and produce more generalized models in order to overcome the overfiting issue.

1. Randomly splits data into k equal sized subsamples
2. Train the model on k-1 subsamples and predict the hold-out kth subsamples
3. Repeated k times, holding out each subsample once as the validation set
4. Evaluate performance by averaging k results to produce a single estimation.

### Regularization
Any modification we make for machine learning algorithms reduce test error but not training error.

**L1 Regualization (Lasso)**: Least Absolute Shrinkage and Selection Operator.
1. Shrinks the less important feature’s coefficient to zero, producing sparse solutions
2. Performs feature selection in case a large set of features.
3. Gradient of Lasso $\frac{dL_1}{dw} = \{\pm1\}$ expect when $W = 0$, which is fixed weight decay. It moves any weights towards 0 with the same step size, regardless the weight's value.
4. Slow converge

**L2 Regularization (Ridge)**: Adds “squared magnitude of the coefficient" as penalty term to the loss function. The regularization parameter
$\lambda$ penalizes all the parameters so that the model generalizes the data and overcome overfit.

1. $\lambda$ is very large, then lead to underfitting
2. Forces the weights to be small but does not make them zero and does not give the sparse solution
3. Gradient of Ridge $\frac{dL_2}{dw} = w$, which is linearly decay. It moves any weight towards 0, but it will take smaller and smaller steps as a weight approaches 0.
4. Solves multicollinearity issue
5. Not robust to outliers as square terms blow up the error differences of the outliers
6. Performs better when all the input features influence the output, and all with weights are of roughly equal size

### Principal Component Analysis (PCA)
An orthogonal linear transformation, "principal components" that transfers the data to a new coordinate system such that the greatest variance by any projection of data. It can be used to reduce dimensions of data without much loss of information.

1. Centerize the data
2. Calculate covariance matrix $C = X^TX$ 
3. Compute the unit eigenvectors $V$ of $C$ by SVD
4. Sort eigenvalues and select top K principle components
5. Transform data to new system: $XV$

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

### Linear Regression
A linear approach to modeling the relationship between a continous response and one or more explanatory variables (predictors).

Assumption:
1. linear relationship between predictors and target variable
2. Gausian noise: error are independent, constant variance
3. No multicollinearity

### Logistic Regression
A statistical model that use the logistic sigmoid function to model a binary label. It constructs a model which estimates probability of each class.

1. Sigmoid function: $\sigma(x) = \frac{1}{1+e^{-x}}$

### Support Vector Machine (SVM)
Construct a optimal hyperplane such that maximize the distance (margin) to the cloest point in the each class.

**Kernel Trick**: Applying the kernel transformation (non-linear function) to all inputs, mapping all data into high-dimensional space, then learning a linear model in the new transformed space.

### Multi-Class Classification
**One vs. One**:

**One vs. All**:

**Softmax**: Takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one, which represent normalized probability distribution over n different classes.

$P(y_i|x) = \frac{e^x}{\sum_i^n e^x}$


<end>



