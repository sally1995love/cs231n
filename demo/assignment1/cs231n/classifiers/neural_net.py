import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    n1 = input_size * hidden_size
    n2 = hidden_size * output_size
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size) * np.sqrt(2.0/n1)  #(D, H)  D=3072 dimensions
    #self.params['W1'] = std * np.random.randn(input_size, hidden_size)  #(D, H)  D=3072 dimensions
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size) * np.sqrt(2.0/n2)
    #self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0, p=0.5):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # input - fully connected layeprint X.shape
    H1 = np.dot(X,W1)  # X(N, D)   W1(D, H)    -> H1(N, H)    b->(H, )
    H1 = H1 + b1  # col broadcast
    # - ReLU - fully connected layer
    H1[np.where(H1<0)] = 0
    #U1 = (np.random.rand(*H1.shape) < p) / p
    #H1 *= U1
    # input - fully connected layer
    H2 = H1.dot(W2)   #H1 (N, H)  W2(H, C)  -> H2 (N, C)
    H2 = H2 + b2
    #U2 = (np.random.rand(*H2.shape) < p) / p
    #H2 *= U2
    scores = H2
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    scores = scores.T
    scores -= np.max(scores) # scores.shape (N, C)    avoid explode
    scores = scores.T
    exp_scores = np.exp(scores)
    exp_scores_sum = np.sum(exp_scores,axis=1) #sum exp^j ( j->category )  (N, )
    exp_correct_scores = np.exp(scores[range(len(y)), y]) # N,
    probs = exp_scores.T / exp_scores_sum  # normalization (C, N)
    loss = -np.mean(np.log(exp_correct_scores / exp_scores_sum)) # N, * N,  cal mean
    loss += 0.5 * reg * np.sum(W1 * W1)
    loss += 0.5 * reg * np.sum(W2 * W2)
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dot1 = np.dot(X,W1)#dot1 (N,H)
    # gradient of RELU of output
    dscores = probs.copy() # C,N
    dscores[y,range(len(y))] -= 1
    dW2 = np.dot(dscores,H1) #  (C, N) H1( N,H)     dW2(C, H)  probs is (softmax)'scores  H1 is input
    dW2 /= X.shape[0] # /N
    grads['W2'] = dW2.T + reg * W2 # W2( H,C)  grads['W2'](H, C)
    grads['b2'] = np.sum(dscores, axis=1) / N

    dRelu = dot1.copy()
    dRelu[dRelu >= 0] = 1
    dRelu[dRelu < 0] = 0
    dH1 = (np.dot(dscores.T, W2.T))  # N,C C,H  N,H
    dH1 = dH1 * dRelu#N,H
    dW1 = np.dot(dH1.T, X) / X.shape[0]#      H, N  N, D  H, D
    grads['W1'] = dW1.T + reg * W1
    grads['b1'] = np.sum(dH1 * dRelu, axis=0) / N
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads
  def update_parameter(self,v,mu,dx,x,learning_rate):
    v_prev = v  # back this up
    v = mu * v - learning_rate * dx  # velocity update stays the same
    # update = -mu * v_prev + (1 + mu) * v
    # update_scale = np.linalg.norm(update.ravel())
    # param_scale = np.linalg.norm(x.ravel())
    # print update/param_scale
    x += -mu * v_prev + (1 + mu) * v  # position update changes form
    return x

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False, mu=0.9,p=0.5):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    v={} # initialize velocity
    v['W1'] = 0
    v['b1'] = 0
    v['W2'] = 0
    v['b2'] = 0
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      indexes = np.random.choice(num_train,batch_size)
      X_batch = X[indexes]
      y_batch = y[indexes]
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the currentf minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg,p=p)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] = self.update_parameter(v['W1'],mu,grads['W1'],self.params['W1'],learning_rate)
      self.params['W2'] = self.update_parameter(v['W2'],mu,grads['W2'],self.params['W2'],learning_rate)
      self.params['b1'] = self.update_parameter(v['b1'],mu,grads['b1'],self.params['b1'],learning_rate)
      self.params['b2'] = self.update_parameter(v['b2'],mu,grads['b2'],self.params['b2'],learning_rate)
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    scores = self.loss(X)
    y_pred = np.argmax(scores,axis=1)
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


