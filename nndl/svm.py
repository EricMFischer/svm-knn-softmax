import numpy as np
import pdb

"""
This code was based off of code from cs231n at Stanford University, and modified for ece239as at UCLA.
"""
class SVM(object):

  def __init__(self, dims=[10, 3073]):
    self.init_weights(dims=dims)

  def init_weights(self, dims):
    """
	Initializes the weight matrix of the SVM.  Note that it has shape (C, D)
	where C is the number of classes and D is the feature size.
	"""
    self.W = np.random.normal(size=dims) # (10,3073)

  def loss(self, X, y):
    """
    Calculates the SVM loss.
  
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
  
    Inputs:
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
  
    Returns a tuple of:
    - loss as single float
    """
  
    # compute the loss and the gradient
    num_classes = self.W.shape[0]
    num_train = X.shape[0] # 49000 (X: (49000,3073), y: (49000,))
    loss = 0.0

    for i in np.arange(num_train):
    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the normalized SVM loss, and store it as 'loss'.
    #   (That is, calculate the sum of the losses of all the training 
    #   set margins, and then normalize the loss by the number of 
    #   training examples.)
    # ================================================================ #
      scores = self.W.dot(X[i].T) # (10,1)
      clf_idx = y[i]
      for j, score in enumerate(scores):
            if j != clf_idx: # if index of score != correct classification index
                loss += max(0, 1 + scores[j] - scores[clf_idx])
    loss /= num_train
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss
  
  def loss_and_grad(self, X, y):
    """
	Same as self.loss(X, y), except that it also returns the gradient.

	Output: grad -- a matrix of the same dimensions as W containing 
		the gradient of the loss with respect to W.
	"""
  
    # compute the loss and the gradient
    num_train = X.shape[0] # 49000 (X: (49000,3073), y: (49000,))
    loss = 0.0
    grad = np.zeros_like(self.W) # (10,3073)

    for i in np.arange(num_train):
    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the SVM loss and the gradient.  Store the gradient in
    #   the variable grad.
    # ================================================================ #
      scores = self.W.dot(X[i].T) # (10,3073)x(3073,1) = (10,1)
      clf_idx = y[i]
      # bad_hyperplanes: classes that contributed 1 or more to loss function,
      # as they inaccurately contained the data point X[i]
      num_bad_hyperplanes = 0

      for j, score in enumerate(scores):
            score_error = score - scores[clf_idx]
            bad_hyperplane = score_error > 0
            num_bad_hyperplanes += 1 if bad_hyperplane else 0
            if j != clf_idx: # if score index != correct classification index
                loss += max(0, 1 + score_error)
                grad[j] += X[i] if bad_hyperplane else 0
      grad[clf_idx] += -1 * num_bad_hyperplanes * X[i]

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    loss /= num_train
    grad /= num_train

    return loss, grad

  def grad_check_sparse(self, X, y, your_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in these dimensions.
    """
  
    for i in np.arange(num_checks):
      ix = tuple([np.random.randint(m) for m in self.W.shape])

      oldval = self.W[ix]
      self.W[ix] = oldval + h # increment by h
      fxph = self.loss(X, y)
      self.W[ix] = oldval - h # decrement by h
      fxmh = self.loss(X,y) # evaluate f(x - h)
      self.W[ix] = oldval # reset
  
      grad_numerical = (fxph - fxmh) / (2 * h)
      grad_analytic = your_grad[ix]
      rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
      print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

  def fast_loss_and_grad(self, X, y):
    """
    A vectorized implementation of loss_and_grad. It shares the same
	inputs and ouptuts as loss_and_grad.
    """
    grad = np.zeros(self.W.shape) # initialize the gradient as zero
    num_train = X.shape[0]

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the SVM loss WITHOUT any for loops.
    # ================================================================ #

    scores = X.dot(self.W.T) # (500,10)
    # selects the correct class scores across all examples
    y_i = np.reshape(scores[np.arange(num_train), y], (-1,1)) # (500,1)
    losses = np.maximum(0, 1 + scores - y_i) # (500,10)
    
    # adjustment: loss summation includes all class indices j except y_i,
    # so for the correct classification index we set the loss to 0
    losses[np.arange(num_train), y] = 0
    loss = np.mean(np.sum(losses, axis=1))
    
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    # ================================================================ #
    # YOUR CODE HERE:
    #   Calculate the SVM grad WITHOUT any for loops.
    # ================================================================ #

    binary_losses = losses # (500,10)
    # set losses greater than 0 to 1, for later multiplication by X
    # (we already set losses for the correct classification index to 0)
    binary_losses[losses > 0] = 1
    num_bad_hyperplanes = np.sum(binary_losses, axis=1) # (500,)
    
    # for the correct classification index, set the loss to -1 * the number
    # of classes with bad hyperplanes for that example. Note: the correct class
    # y_i won't be mistakenly included in num_bad_hyperplanes, because in
    # the computation of 'losses' above, we set the loss for the correct
    # classification index y_i to 0
    binary_losses[np.arange(num_train), y] = -1 * num_bad_hyperplanes
    grad_term = X.T.dot(binary_losses) # X.T: (3073,500), grad_term: (3073,10)
    grad = grad_term.T / num_train

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return loss, grad

  def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
    # X examples = 49000, batch_size = 200, num_iters = 1500
    num_train, dim = X.shape # (49000, 3073), y: (49000,)
    num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes (10)

    self.init_weights(dims=[num_classes, X.shape[1]]) # initializes the weights of self.W (10, 3073)

    # Run stochastic gradient descent to optimize W
    loss_history = []

    for it in np.arange(num_iters): # 1500 iterations
      X_batch = None
      y_batch = None

      # ================================================================ #
      # YOUR CODE HERE:
      #   Sample batch_size elements from the training data for use in 
      #   gradient descent.  After sampling,
      #     - X_batch should have shape: (dim, batch_size)
      #     - y_batch should have shape: (batch_size,)
      #   The indices should be randomly generated to reduce correlations
      #   in the dataset.  Use np.random.choice.  It's okay to sample with
      #   replacement.
      # ================================================================ #
      # X_batch: (3073, 200), y_batch: (200,)
      
      batch_idxs = np.random.choice(np.arange(num_train), batch_size, replace=True)
      X_batch = X[batch_idxs] # (200,3073)
      y_batch = y[batch_idxs] # (200,)

      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      # evaluate loss and gradient
      loss, grad = self.fast_loss_and_grad(X_batch, y_batch) # grad: (10,3073)
      loss_history.append(loss)

      # ================================================================ #
      # YOUR CODE HERE:
      #   Update the parameters, self.W, with a gradient step 
      # ================================================================ #
      self.W -= learning_rate * grad
      # ================================================================ #
      # END YOUR CODE HERE
      # ================================================================ #

      if verbose and it % 100 == 0:
        print('iteration {} / {}: loss {}'.format(it, num_iters, loss))

    return loss_history

  def predict(self, X):
    """
    Inputs:
    - X: N x D array of training data. Each row is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
    # X: (49000, 3073) for training, (1000, 3073) for validation data
    y_pred = np.zeros(X.shape[0]) # (49000,)

    # ================================================================ #
    # YOUR CODE HERE:
    #   Predict the labels given the training data with the parameter self.W.
    # ================================================================ #
    y_pred = np.argmax(X.dot(self.W.T), axis=1) # (49000,10)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return y_pred

