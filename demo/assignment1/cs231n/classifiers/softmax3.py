import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  W = W.transpose()
  X = X.transpose()
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(X.shape[1]): # for each sample
    scores = W.dot(X[:,i]) # scores:10*1
    scores -= np.max(scores)# must be put ahead of the 'correct_class_score = scores[y[i]]'
    correct_class_score = scores[y[i]]
    sum_prob = np.sum(np.exp(scores))
    prob = np.exp(correct_class_score) / sum_prob
    loss -= np.log(prob)
    for j in xrange(scores.shape[0]):
      prob_ji = np.exp(scores[j]) / sum_prob
      if j==y[i]:
        margin = (1 - prob_ji) * X[:, i].T
      else:
        margin = -prob_ji*X[:,i].T
      dW[j,:] -= margin
  loss /= X.shape[1]
  dW /= X.shape[1]
  loss += 0.5*reg*np.sum(W*W)
  dW += reg*np.sum(W*W)
  pass
  dW = dW.T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  # loss = 0.0
  W = W.transpose()
  X = X.transpose()
  # dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = W.dot(X)
  scores -= np.max(scores,axis=0)
  Loss = -scores[y,range(len(y))] + np.log(np.sum(np.exp(scores),axis=0))
  loss = np.mean(Loss)
  loss += 0.5 * reg * np.sum(W * W)
  exp_scores = np.exp(scores) # 10*500
  exp_sum = np.sum(exp_scores,axis=0) # 1*500
  n_probs = -exp_scores / exp_sum # - prob_ji * X[:, i].T
  n_probs[y,range(len(y))] += 1 # (1 - prob_ji) * X[:, i].T
  dW = n_probs.dot(X.T) # 10*500 & 500*3073 -> 10*3073
  dW /= -X.shape[1]
  dW += reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW = dW.T
  return loss, dW

