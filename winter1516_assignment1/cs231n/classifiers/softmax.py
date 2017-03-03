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
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N=X.shape[0]
  C=W.shape[1]
  for i in range(N):
    scores=X[i]@W
    maxScore=np.max(scores)
    expScores=np.exp(scores-maxScore)
    expScoresSum=np.sum(expScores)
    loss-=np.log(expScores[y[i]]/expScoresSum)
    dW[:,y[i]]-=X[i]
    for j in range(C):
      dW[:,j]+=X[i]*expScores[j]/expScoresSum
  loss/=N
  dW/=N
  loss+=0.5*reg*np.sum(W**2)
  dW+=reg*W
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
  loss = 0.0
  dW = np.zeros_like(W)
  N=X.shape[0]
  D=X.shape[1]
  C=W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores=X@W
  maxScore=np.max(scores,1).reshape(N,1)
  expScores=np.exp(scores-maxScore)
  expScoresSum=np.sum(expScores,1).reshape(N,1)
  indicator=np.zeros_like(scores)
  indicator[range(N),y]=1
  loss-=np.sum(np.log(np.sum(expScores*indicator,1).reshape(N,1)/expScoresSum))
  dW-=X.T@indicator
  dW+=X.T@(expScores/expScoresSum)
  loss/=N
  dW/=N
  loss+=0.5*reg*np.sum(W**2)
  dW+=reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

