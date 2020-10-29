import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# X coordinates of points used for the Bias-Variance trade-off example.
BV_PTS_X = [-0.06, 1.83, 1.97, 3.48, 3.30]
BV_PTS_X_NEW = BV_PTS_X + [2.5]

# Y coordinates of points used for the Bias-Variance trade-off example.
BV_PTS_Y = [0.39, 1.68, 2.73, 2.85, 4.28]
BV_PTS_Y_NEW = BV_PTS_Y + [2.65]

def _bv_get_empty_chart():
  """Gets an empty chart to plot the Bias-Variance trade-off example charts
  inside it."""

  fig, ax = plt.subplots()
  ax.set_xlim(-1,6)
  ax.set_ylim(-1,6)

  return fig, ax

def _bv_plot_linear_regression(x, y, ax):
  """Plots a linear regression line for points (x,y) on a given pyplot ax.

  # Arguments

    x: int[], vector with the independent variable values
    y: int[], vector with the dependent variable values
    ax: pyplot ax, where to plot the regression line

  # Returns

    A pair with the pyplot plotted object, and the R^2 score of the regression.
  """

  X = np.reshape(x, (-1, 1))
    
  model = LinearRegression()
  model.fit(X,y)
    
  intercept = model.intercept_
  slope = model.coef_[0]
    
  regression_line, = ax.plot(
    [-1, 6],
    [intercept + -1 * slope, intercept + 6 * slope]
  )
    
  return regression_line, model.score(X, y)

def _bv_plot_poly_regression(x, y, degree, ax):
  """Plots a polynomial regression line for points (x,y) on a given pyplot ax.

  # Arguments
  
    x: int[], vector with the independent variable values
    y: int[], vector with the dependent variable values
    degree: int, degree of the polynomial
    ax: pyplot ax, where to plot the regression line

  # Returns
  
    A pair with the pyplot plotted object, and the R^2 score of the regression.
  """

  X = np.reshape(x, (-1, 1))
    
  model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
  model.fit(X,y)
    
  poly_x = np.linspace(-1, 6, 100)
  poly_y = model.predict(poly_x.reshape(-1, 1))
    
  regression_curve, = ax.plot(poly_x, poly_y)
    
  return regression_curve, model.score(X, y)

def bv_chart_five_pts():
  """Plots the initial five points in the training set for the Bias-Variance
  trade-off example."""

  fig, ax = _bv_get_empty_chart()
  ax.scatter(BV_PTS_X, BV_PTS_Y)
  ax.set_title("Initial training set with five points")

  return fig, ax

def bv_chart_six_pts():
  """Plots the augmented trainign set, with 6 points, for the Bias-Variance
  trade-off example."""

  fig, ax = _bv_get_empty_chart()
  ax.scatter(BV_PTS_X_NEW, BV_PTS_Y_NEW)
  ax.set_title("New training set with six points")

  return fig, ax

def bv_chart_linear():
  """Plots the trained linear model over the five points in the training set
  for the Bias-Variance trade-off example."""

  fig, ax = bv_chart_five_pts()
  _, score = _bv_plot_linear_regression(BV_PTS_X, BV_PTS_Y, ax)
  ax.set_title(
    "Trained linear model on a set with five points. "
    r"$R^2$: {:.3f}".format(score))

  return fig, ax

def bv_chart_poly():
  """Plots the trained polynomial model over the five points in the training 
  set for the Bias-Variance trade-off example."""

  fig, ax = bv_chart_five_pts()
  _, score = _bv_plot_poly_regression(BV_PTS_X, BV_PTS_Y, 4, ax)
  ax.set_title(
    "Trained polynomial model on a set with five points. "
    r"$R^2$: {:.3f}".format(score))

  return fig, ax

def bv_chart_linear_new():
  """Plots the trained linear model over the six points in the augmented
  training set for the Bias-Variance trade-off example."""

  fig, ax = bv_chart_six_pts()
  before, _ = _bv_plot_linear_regression(BV_PTS_X, BV_PTS_Y, ax)
  after, score = _bv_plot_linear_regression(BV_PTS_X_NEW, BV_PTS_Y_NEW, ax)
  ax.set_title(
    "Trained linear model on the augmented set with six points. "
    r"$R^2$: {:.3f}".format(score)
  )
  fig.legend(
        [before, after],
        ['Linear regression (before)', 'Linear regression (after)'],
        loc = 'center left',
        bbox_to_anchor = (0.55, 0.15)
    )

  return fig, ax

def bv_chart_poly_new():
  """Plots the trained polynomial model over the six points in the augmented
  training set for the Bias-Variance trade-off example."""

  fig, ax = bv_chart_six_pts()
  before, _ = _bv_plot_poly_regression(BV_PTS_X, BV_PTS_Y, 4, ax)
  after, score = _bv_plot_poly_regression(BV_PTS_X_NEW, BV_PTS_Y_NEW, 4, ax)
  ax.set_title(
    "Trained polynomial model on the augmented set with six points. "
    r"$R^2$: {:.3f}".format(score)
  )
  fig.legend(
        [before, after],
        ['Polynomial regression (before)', 'Polynomial regression (after)'],
        loc = 'center left',
        bbox_to_anchor = (0.55, 0.15)
    )

  return fig, ax

def chart_square_loss():
  """Plots a chart of the square loss for a true Y value of 1."""

  yhat = np.linspace(-1, 3, 100)
  y = [1] * len(yhat)

  plt.plot(yhat, (yhat - y) ** 2, c = 'b')
  plt.plot([1, 1], list(plt.ylim()), '--', c = 'r')
  plt.xlabel(r"$\hat{Y}$")
  plt.ylabel(r"$\ell(1, \hat{Y})$ = square loss")

def chart_binary_loss():
  """Plots a chart of the binary loss for a true Y value of 1."""
  yhat1 = [-1, 0.96]
  yhat2 = [1.04, 3]

  plt.plot(yhat1, [1] * len(yhat1), c = 'b')
  plt.plot(yhat2, [1] * len(yhat2), c = 'b')
  plt.plot([1], [1], c = 'b', marker = 'o', markersize = 10,
    markerfacecolor = 'none', markeredgewidth = 3)
  plt.plot([1], [0], c = 'b', marker = 'o', markersize = 10,
    markeredgewidth = 3)
  plt.plot([1, 1], list(plt.ylim()), '--', c = 'r')
  plt.xlabel(r"$\hat{Y}$")
  plt.ylabel(r"$\ell(1, \hat{Y})$ = binary loss")

def chart_hinge_loss():
  """Plots a chart of the hinge loss for a true Y value of 1."""

  yhat = np.linspace(-1, 3, 200)
  y = [1] * len(yhat)

  zero = [0] * len(yhat)
  one = [1] * len(yhat)

  plt.plot(yhat, np.maximum(zero, one - y * yhat))
  plt.plot([1, 1], list(plt.ylim()), '--', c = 'r')
  plt.xlabel(r"$\hat{Y}$")
  plt.ylabel(r"$\ell(1, \hat{Y})$ = hinge loss")