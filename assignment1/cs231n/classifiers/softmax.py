import numpy as np
from random import shuffle
from past.builtins import xrange

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
  dW = np.zeros_like(W)
  N = X.shape[0]
  K = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  for i in range(N):
    scores[i] -= max(scores[i])
    sum_exp_scores = np.sum(np.exp(scores[i]))
    exp_score = np.exp(scores[i, y[i]])
    loss_i = -np.log(exp_score / sum_exp_scores)
    loss += loss_i
    for j in xrange(K):
      prob_ji = np.exp(scores[i, j]) / sum_exp_scores
      if j==y[i]:
        margin = (1 - prob_ji) * X[i,:]
      else:
        margin = -prob_ji*X[i,:]
      dW[:,j] -= margin
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= N
  loss += reg * np.sum(W * W) * 0.5
  dW /= N
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = X.shape[0]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  max_scores = np.max(scores, axis = 1, keepdims=True) #(N,1)
  scores -= max_scores
  correct_exp_scores = np.exp(scores[range(N), y]) #(N,1)
  sum_exp_scores = np.sum(np.exp(scores), axis=1, keepdims=True) # (N,1)
  loss = -np.mean(np.log(correct_exp_scores / sum_exp_scores))
  loss += 0.5 * reg * np.sum(W * W)

  probs = np.exp(scores) / sum_exp_scores #(N,K)

  probs[range(N), y] -= 1
  dW = X.T.dot(probs)
  dW /= N
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

