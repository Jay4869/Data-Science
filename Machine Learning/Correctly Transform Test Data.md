# Correctly Transform Test Data

Many machine learning algorithms require that features are on the same scale; for example, if we compute distances such as in nearest neighbor algorithms. Also, optimization algorithms such as gradient descent work best if our features are centered at mean zero with a standard deviation of one. For example, the data has the properties of a standard normal distribution. One of the few categories of machine algorithms that are truly scale invariant are the tree-based methods.

### Scenario 1
scaled_data = (data - data_mean) / data_sd
train, test = split(scaled_data)

### Scenario 2
train, test = split(data)
scaled_train =  (train - train_mean) / train_sd
scaled_test = (test - test_mean) / test_sd

### Scenario 3
scaled_train =  (train - train_mean) / train_sd
scaled_test = (test - train_mean) / train_sd


The correct way is **Scenario 3**! The reason is that we want to pretend that the test data is "new, unseen data". We use the test dataset to get a good estimate of how our model performs on any new data. In sum, if we standardize our training dataset, we need to keep the parameters (mean and standard deviation for each feature). Then, we’d use these parameters to transform our test data and any future data later on.

In the **Scenario 1**, we need split our data to training set and test set first because we don't want test data to be a part of data processing or model training step, which it makes test data non-generalized to the real world and causes overfitting.

In the **Scenario 2**, if you are going to scale test data with itself, then you assume that trianing and test data are following the same distribution, but it might not be true! Therefore, incorrect scaling will effect on the tuning hyper-parameters of models. In a real application, the new, unseen data could be just 1 data point that we want to classify. (How do we estimate mean and standard deviation if we have only 1 data point?) That’s an intuitive case to show why we need to keep and use the training data parameters for scaling the test set.