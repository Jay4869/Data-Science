# Deep Learning Concepts

### Multi-layer Perceptrons (MLP)
Multi-layer Perceptrons are multi-input and multi-ouput parameteric functions, composing together.
1. Each sub-function is a layer of network, and each layer output of functions is a unit/feature
2. Width: number of units in each layer
3. Depth: number of layers

### Activation Functions
Control output value produced by a neuron and decide whether outside connections should consider this neuron as “actived” or not

**Identity/Linear**: predict continuous target values using a linear combination of signals that arise from one or more layers of nonlinear transformations of the input.

**Sigmoid (Logistic) Function**: a soft assignment of step function as the probability of an artificial neuron “firing” given its inputs.
$$\sigma(x) = \frac{1}{1+e^{-x}}\\
\partial\sigma(x) = \sigma(x)(1-\sigma(x))$$
1. Nonlinear in nature
2. range between (0,1)
3. Small change input, large change output
4. When towards the end, the output tend to respond very less to changes in X, so it will cause vanishing gradients

**Hyperbolic Tangent**: Strongly negative inputs to the tanh will map to negative outputs. Additionally, only zero-valued inputs are mapped to near-zero outputs.
$$f(x) = tanh(x) = \frac{e^z - e^{-z}}{e^z + e^{-z}}\\
\partial f(x) = 1-f^2(x)$$

**ReLu**: Rectified Linear Unit
$$\phi(x) = max(0, x)$$

**Softmax**:
$$\phi(x) = \frac{e^x}{\sum_i^n e^x}$$
For small dataset, it is even more important than learning the weights of the hidden layers.

**Softplus**:
$$\phi(x) = log(1+e^x)$$

**Hard tanh**:
$$\phi(x) = max(-1, min(1,x))$$

**Absolute Value Rectification**:
$$\phi(x) = |x|$$

**Maxout**:

### Artificial Neural Networks (ANN)
1. A feedforward network provide a universial system for representing functions, but there is no guarantees that the training algorithm will be able to learn it.
2. Approximate any functions mapping from any finite dimensional discrete space to another
3. two challenge: optimization and overfiting
4. Consist of two piecewise linear propagation and weight vector


## Optimization
**Unconstrained Problem**: Apply the gradient-based optimization, and the standard steepest descent algorithm: $x \leftarrow x - \eta\ \nabla f(x)$

**Constrained Problem**: Lagrangin

* In low-dimensioanl space, local minima are common
* In higher dimensional spaces, local minima are rare and saddle points are exponetially more common

**Empirical Risk Minimization**: The expectation is taken over the true underlying distribution, so the risk is a form of generalization error. Since we know the distribution from training set, it would be solved by an optimization algorithm.
1. Overfitting: model with high capacity can simply memorize the training set.
2. Some loss functions are not feasible such as 0-1 loss.

### Gradient Descent
The algorithm calls for updating the model parameters (weights and biases) with a small step in the direction of the gradient of the objective function that includes the terms of all the training set.

The batch size is a hyper-parameter that defines the number of samples to work through before updating the internal model parameters. Think of a batch as a resampling method over one or more samples to train the model. At the end of the batch, the predictions are compared to the expected output variables and calculated the error. Then, perform Back-Propagation to improve the model.

1. **Batch Gradient**: use the entired training set and guarantee to reduce the loss with small fluctuation. For every new input data, all of data are reused for optimization.
2. **Stochastic Gradient**: use only a single of data point at a time with large fluctuation. Only the new input data is used for optimization.
3. **Mini-Batch Gradient**: use a subset of data and reduce the variance of parameter updates compared to SGD.
    * Larger bactches provide more accuracy estimate of the gradient, but less than SGD
    * Smaller batches offer a regularizing effect, but time consuming.
    * Multicore GPU architectures are designed by extremely small batches. It is common for power of 2 batch sizes to offer best runing time.


### Epoch
The number of epochs is also a hyper-parameter that defines the number times that the learning algorithm will work through the entire training dataset. For exmaple, One epoch is comprised of one or more batches, and each batch in the training dataset has an opportunity to update the internal model parameters. It is common to create line plots that show epochs along the x-axis as time and the error or skill of the model on the y-axis. These plots are sometimes called learning curves. These plots can help to diagnose whether the model has over learned, under learned, or is suitably fit to the training dataset.

* batch size is a number of samples processed before the model is updated. $(1 <= Batch Size <= training\ size)$
* epoch is the number of passes through the entire training dataset. epochs can be set to an integer value between one and infinity. 


### Learning Rate
An optimization hyperparameter that control the size of the step the parameters take in the direction of gredient. The choice of learning rate impacts the ability to avoid numerous suboptimal local minima for highly non-convex error functions.

To converage to a minimum, the learning rate of SGD/Mini-Batch should decrease because the gradient estimator introduces noise when randomly learn from training samples.

* Large learning rate -> fast convergence & high noise
* Small learning rate -> slow convergence & low noise, local minima possiblity

### Back-Propagation
A gradient descent method based on the chain rule for computing the gradient and updating the weights of the network by moving layer by layer from the final layer and back to the initial layer.

### Gradient Vanishing
When we do Back-propagation, moving backward in the Network and calculating gradients of loss (Error) with respect to the weights, the gradients tend to get smaller and smaller as we keep on moving backward in the Network. After small gradients are multiplied together, the gradient decreases exponentially down to the initial layers. A small gradient means that the weights and biases of the initial layers will not be updated effectively with each training session.

![](https://i.imgur.com/6C2NDo1.png)


### Gradient Exploding
In deep networks or recurrent neural networks, the error gradients can be accumulated during updates, which results in very large gradients and updates to the network weights. At an extreme, the values of weights can become so large as to overflow and result in NaN values. The explosion occurs through exponential growth by repeatedly multiplying gradients through the network layers that have values larger than 1.0.

### Handling the Vanishing Gradient
The simplest solution is to use other activation functions, such as ReLU, which doesn’t cause a small derivative. Gradient or slope of RELU activation if it’s over 0 is 1. Sigmoid derivative has a maximum slope of .25. RELU activation solves this by having a gradient slope of 1, so during back-propagation, there isn’t gradients passed back that are progressively getting smaller and smaller. but instead they are staying the same, which is how RELU solves the vanishing gradient problem.

However, RELU is that if you have a value less than 0, that neuron is dead, and the gradient passed back is 0, meaning that during back-propagation, you will have 0 gradient being passed back if you had a value less than 0. An alternative is Leaky RELU, which gives some gradient for values less than 0 in order to solve dead neuron issue and speed up. (it helps keep off-diagonal entries of the Fisher information matrix small)

## Optimizer in Deep Learning
**SGD Momentum**: Momentum accelerates SGD to solve poor conditioning of the Hessian matrix and vairance in SGD. We introduce a variable that is the direction and speed at which the parameteres move through the parameter space. The momentum is set to an exponentially decaying average of the negative gradient.

**Nesterov momentum**: It is attempting to add a correction factor to the gradient in the standard momentum method, which update the gradient after the current momentum is applied.

**AdaGrad**: Individually adapts the learning rate for each model parameter, by scaling it inversely proportional to an accumulated sum of squared partial derivatives over all training iterations.

The parameters with the largest partial derivative of the loss have a correspondingly rapid decrease in their learning rate, while parameters with small partial derivatives have a relatively small decrease in their learning rate.

1. Dealing with sparse data improving the robustness of SGD
2. Eliminate manual with tuning of the learning rate
3. Accumulation of the squared gradients might be very small.

**Adadelta**: an extension of Adagrad that the sum of gradient as a decaying average of all past squared gradients.


**RMS Prop**: The RMSprop algorithm addresses the deficiency of AdaGrad by changing the gradient accumulation into an exponentially weighted moving average. The introduction of the exponentially weighted moving average allows the effective learning rates to adapt to the changing local loss.

**Adam**: Adam is a variant on RMSprop+momentum
1. Rescale gradients by the first order moment (mean) and second order moment (squared gradient) with exponentially decay
2. Includes bias corrections to the estimates of both the first-order moments (mean) and the (uncentered) second order moments to account for their initialization at the origin.


## Neural Network Techniques
### Number of Neurons
Using too few neurons in the hidden layers will result in something called underfitting. Underfitting occurs when there are too few neurons in the hidden layers to adequately detect the signals in a complicated data set.

Using too many neurons in the hidden layers may result in overfitting. Overfitting occurs when the neural network has so much information processing capacity, but the training set is not enough to train all of the neurons in the hidden layers. A second problem can occur even when the training data is sufficient. An inordinately large number of neurons in the hidden layers can increase the time it takes to train the network.

* The number of hidden neurons should be between the size of the input layer and the size of the output layer.
* The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
* The number of hidden neurons should be less than twice the size of the input layer.

### Pooling
A pooling function replaces the output of the layer with a summary local statistic. Pooling can make the representation become invariant to small local translations of the input.
1. Allow flexibility in the location of certain features
2. Union similiar features together
3. Reduce the output image size

### Stride
Jump the filtering process to smooth the same information repeatly

### Zero Padding
An essential feature of any convolutional network implementation is the ability to implicitly zero-pad the input in order to make it wider. Zero padding the input allows us to control the kernel width and the size of the  output independently.
1. filtering reduces spatial extent of feature map
2. Edge pixels are less frequently seen

### Injecting Noise/Adversarial Training
Either add noise to the inputs or inject noise into the weights uring training. 
1. Build in some local robustness into the model and thereby promote generalization.
2. Push the model parameters into the region where the model is relatively insensitive to small variations in the weights.
3. In Neural networks, add a small noise to the inputs as the linear building blocks $\hat{y}=w^Tx+w^T\varepsilon$

### Early Stopping
A training procedure like gradient descent will tend to learn more and more complex functions as the number of iterations increases. By regularization on time, the complexity of the model can be controlled, improving generaliztion.

Instead of running our optimization algorithm until reaching a minimum of validation error, we store the model parameters every time the validation error improves, and run until the error has not improved.

### Data Augmentation
Operate the training images pixels in each direction can often improve generalization.

### Bootstrap Aggregation (Bagging)
1. Train a model by random sampling with replacement. 
2. Repeat k times to produce k separate models
3. All models vote on output for test samples
4. Ensemble method requires long training time and momery, not friendly for scaling.

### Dropout
Thinking of dropout as a form of regularization, at each training stage, the neuron network is randomly excluded from node activation and weight updates by given probability.

1. Dropout forces a neural network to learn more robust features on many different random subsets of the other neurons to reduce overfitting
2. Dropout roughly doubles the number of iterations required to converge. However, training time for each epoch is less.
3. The removed nodes are reinserted into the network with their orignal weights.
4. More effective than other standard computationlly regularizations.
5. For input nodes, dropout should be lower because information is explicitly lost when input nodes are ignored.

### DropConnect
The generalization of dropout in which each connection rather than each output neuron. As dropout, it introduces dynamic sparsity within the model, but differs in that the sparsity in on the weights, rather than the output layer.


## Convolutional Neural Network
A neural network with some number of convolutional layers for modern computer vision.

### Convolution Operation
The convolution is defined by $s(t) = \int x(a)w(t-a)da$, where $x$ denotes the input, $w$ denotes the kernel, and $s$ is feature map. In the multi-dimensional case, the kernel is usually a multi-dimensional tensor.
$s[i,j]=(I*K)[I,J]=\sum_m \sum_n I[m,n]K[i-m,j-n]$

1. sparse connectivity
2. parameter sharing
3. equivariant representation
4. a linear operation on the whole depth of the input. 1x1 conv filter -> dimension reduction

## RestNet
RestNet is a network-in-network architectures which uses micro-architecture models. The micro-architecture models means the set of 'building blocks' to construct the network. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.

The formulation $F(x)+x$ can be realized by feedforward neural networks with identity mapping that are skipping one or more layers.
1. forward: $X_L= X_l + \sum_l^{L-1} F(x_i)$
2. backward: $\frac{\partial E}{\partial x_l} = \frac{\partial E}{\partial x_L}(1+\frac{\partial }{\partial x_l}\sum_l^{L-1}F(x_i))$
3. The model size is actually substantially smaller due to the usage of global average pooling rather than fully-connected layers, down to 102MB for ResNet50

## Recurrent Neural Network (RNN)
RNN is a family of neural networks for processing sequential data that use their internal state (memory) to process sequences of the entire history of previous inputs in each output.

1. A single time step of the input is provided to the network.
2. Then calculate its current state using set of current input and the previous state.
3. The current ht becomes $h_{t-1}$ for the next time step.
4. One can go as many time steps according to the problem and join the information from all the previous states.
5. Once all the time steps are completed the final current state is used to calculate the output.
6. The output is then compared to the actual output i.e the target output and the error is generated.
7. The error is then back-propagated to the network to update the weights and hence the network (RNN) is trained.

## Long Short-Term Memory (LSTM)
LSTM is a special kind of RNN, capable of learning long-term dependencies. LSTMS are explicitly designed to avoid the long-term dependency problem. Remembering information for long periods of time is practically their default behavior, not something they struggle to learn.

### Cell State
* The key to LSTMs is the cell state, the horizontal line running through the top of the diagram.
* The cell state is kind of like a conveyor belt. It runs straight down the entire chain, with only some minor linear interactions. It’s very easy for information to just flow along it unchanged.
* The LSTM does have the ability to remove or add information to the cell state, carefully regulated by structures called gates.

### Gating Mechanism
* Gates are a way to optionally let information through. They are composed out of a sigmoid neural net layer and a point wise multiplication operation.
* The sigmoid layer outputs numbers between 0 and 1, describing how much of each component should be let through. A value of 0 means “let nothing through”, while a value of 1 means “let everything through”.
* LSTM has 3 gates to control the cell state by keeping or removing information
* The forget gate layer: the first step in LSTM is to decide what information to throw away from the cell state. This decision is made by a sigmoid layer. It looks at h(t-1) and x(t) and outputs a number between 0 an 1 for each number in the cell state C(t-1).
* Input gate layer decides which values we’ll update or what new information we’re going to store. Next, a tanh layer creates a vector of new candidate values C(t) that could be added to the state. In the next step, we’ll combine these two to create an update to the state. We can update the old cell state C(t-1) into new cell state C(t), by multiplying old state by f(t), forgetting the things we decided to forget earlier. The we add i(t) * C’(t), this is the new candidate values, scaled by how much we decided to update each state value.
* Output gate layer: Creating a vector after applying tanh function to the cell state, thereby scaling the values to the range -1 to +1. Making a filter using the values of h_t-1 and x_t, such that it can regulate the values that need to be output from the vector created above. This filter again employs a sigmoid function. Multiplying the value of this regulatory filter to the vector created in step 1, and sending it out as a output and also to the hidden state of the next cell.
![](https://i.imgur.com/bvEiFFe.png)

### Update Cell State
To update the old cell state, C(t-1), into the new cell state C(t), cell state is the LSTMs’s neuron: we multiply the old state by f(t), forgetting the things we decided to forget earlier. Then we add i(t) * C(t). This is the new candidate values, scaled by how much we decided to update each state value.
![](https://i.imgur.com/7GGLKCB.png)

### Avoid Gradient Vanishing
If we have a very long sequence inputs, at some point RNN is gonna running into vanishing gradient.

* There are two factors that affect the magnitude of gradients — the weights and the activation functions (or more precisely, their derivatives) that the gradient passes through. If either of these factors is smaller than 1, then the gradients may vanish in time; if larger than 1, then exploding might happen. For example, the tanh derivative is < 1 for all inputs except 0; sigmoid is even worse and is always < 0.25.
* In the re-currency of LSTMs the activation function is the identify function with a derivative of 1.0. So, the back-propagated gradient neither vanishes or explodes when passing through, but remains constant.
* The effective weight of the re-currency is equal to the forget gate activation. So, if the forge gate is on (activation close to 1.0), then the gradient does not vanish. Since the forget gate activation is never > 1.0, the gradient can’t explode either. So that’s why LSTM is so good at learning long range dependencies.


















