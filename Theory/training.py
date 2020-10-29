import matplotlib.animation as anim
import matplotlib.pyplot as plt

import numpy as np
import itertools
import random

def _gradient_points(
  function,
  function_gradient,
  start_point,
  step_size = 0.1,
  num_iterations = 10
):
  """Gives all the points visited by the Gradient Descent algorithm, for a
  given function and number of iterations.
  
  # Arguments

  function: fn, the function we are trying to minimise
  function_gradient: fn, the gradient of `function`
  start_point: float, initial point for gradient descent
  step_size: float, learning rate of the algorithm
  num_iterations: int, number of iterations of gradient descent

  # Returns

  A pair of lists. The first contains all the points visited, and the
  second the values of the function at these points.
  """

  xs = [start_point]
  ys = [function(start_point)]
    
  for i in range(num_iterations - 1):
    new_x = xs[i] - step_size * function_gradient(xs[i])
    xs.append(new_x)
    ys.append(function(new_x))
        
  return xs, ys

def _flatten_array(ary):
  """Flattens a multi-dimensional array to a 1d one.
  Example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]

  # Arguments

  ary: list, an array

  # Returns

  A flattened version of `ary`.
  """

  return list(itertools.chain.from_iterable(ary))

def _init_graph(lines, points, coords):
  """Initialise lines, points, and coordinates of an empty animated graph."""

  lines.set_data([], [])
  points.set_data([], [])
  coords.set_text('')
  
  return lines, points, coords

def _update_graph(i, xs, ys, lines, points, coords):
  """Update the animated chart's data simulating an iteration of GD."""

  all_pts_x = [xs[j] for j in range(i)]
  all_pts_y = [ys[j] for j in range(i)]
        
  points.set_data(all_pts_x, all_pts_y)

  all_segs_x = _flatten_array([list(np.linspace(xs[j], xs[j + 1], 50)) for j in range(i - 1)])
  all_segs_y = _flatten_array([list(np.linspace(ys[j], ys[j + 1], 50)) for j in range(i - 1)])
    
  lines.set_data(all_segs_x, all_segs_y)
    
  if(i > 0):
    coords.set_text("Current g({:.2f}) = {:.2f}".format(all_pts_x[i - 1], all_pts_y[i - 1]))
    
  return lines, points, coords

def animate_gd(function, function_gradient, **kwargs):
  """Produces an animation of the Gradient Descent algorithm applied to
  a given function.

  # Arguments

  function: fn, the function we are minimising
  function_gradient: fn, the gradient of the function we are minimising
  start_point: float, initial point of the GD algorithm
  step_size: float, learning rate of the GD algorithm
  num_iterations: int, number of GD algorithm iterations
  x_bounds: int[2], left and right bounds of the x axis in the animation

  # Returns

  The animation object, which can be displayed in Jupyter.
  """

  start_point = kwargs.get('start_point', 0.0)
  step_size = kwargs.get('step_size', 0.1)
  num_iterations = kwargs.get('num_iterations', 10) + 1
  x_bounds = kwargs.get('x_bounds', [-2.0, 2.0])
  xs, ys = _gradient_points(function, function_gradient, start_point, step_size, num_iterations)
  fig, ax = plt.subplots(figsize = (10, 10))
  plt.close()
  
  function_x = np.linspace(*x_bounds, 100)
  function_y = [function(x) for x in function_x]
  
  ax.plot(function_x, function_y)
  
  lines, = ax.plot([], [], 'r')
  points, = ax.plot([], [], 'bo')
  coords = ax.text(0.8, 0.05, '')
  
  init_f = lambda: _init_graph(lines, points, coords)
  anim_f = lambda i: _update_graph(i, xs, ys, lines, points, coords)
  
  return anim.FuncAnimation(fig, anim_f, init_func = init_f, frames = num_iterations, interval = 1000, blit = True, repeat = False)

def _sgd_g(x, y):
  """Function to minimise, used in the SGD example."""
  return x**2 / 10 + y**2 / 10 + np.sin(x) * np.sin(y)

def _sgd_g_1_diff(x, y):
  """Gradient of function g_1, used in the SGD example."""
  return [x / 5, 0]

def _sgd_g_2_diff(x, y):
  """Gradient of function g_2, used in the SGD example."""
  return [0, y / 5]

def _sgd_g_3_diff(x, y):
  """Gradient of function g_3, used in the SGD example."""
  return [np.cos(x) * np.sin(y), np.sin(x) * np.cos(y)]

def _sgd_g_diff_x(x, y):
  """Derivative of `_sgd_g` wrt to variable x."""
  return x / 5 + np.cos(x) * np.sin(y)

def _sgd_g_diff_y(x, y):
  """Derivative of `_sgd_g` wrt to variable y."""
  return y / 5 + np.sin(x) * np.cos(y)

def _gradient_points3d(
  start_x,
  start_y,
  step_size = 0.1,
  num_iterations = 100,
  stochastic = False
):
  """Gives all the points visited by the (S)GD algorithm applied to the example
  function `_sgd_g`.
  
  # Arguments

  start_x: float, first coordinate of the initial point
  start_y: float, second coordinate of the initial point
  step_size: float, learning rate of the algorithm
  num_iterations: int, number of iterations of gradient descent
  stochastic: bool, whether to use SGD or simple GD

  # Returns

  A triple of lists. The first two contain all the points visited (the x
  coordinate in the first list, and the y coordinate in the second), and the
  third contains the values of the function at these points.
  """

  xs = [start_x]
  ys = [start_y]
  zs = [_sgd_g(start_x, start_y)]

  random.seed(30061988)

  for i in range(num_iterations - 1):
    rnd = random.random()
    diff_x = 0
    diff_y = 0

    if (not stochastic) or (rnd < 0.33):
      diff_x, diff_y =[d + new_d for d, new_d in \
        zip([diff_x, diff_y], _sgd_g_1_diff(xs[i], ys[i]))]
            
    if (not stochastic) or (0.33 < rnd and rnd < 0.66):
      diff_x, diff_y = [d + new_d for d, new_d in \
        zip([diff_x, diff_y], _sgd_g_2_diff(xs[i], ys[i]))]

    if (not stochastic or 0.66 <= rnd):
      diff_x, diff_y = [d + new_d for d, new_d in \
        zip([diff_x, diff_y], _sgd_g_3_diff(xs[i], ys[i]))]

    new_x = xs[i] - step_size * diff_x
    new_y = ys[i] - step_size * diff_y

    xs.append(new_x)
    ys.append(new_y)
    zs.append(_sgd_g(new_x, new_y))

  return xs, ys, zs

def _init_graph3d(lines, s_lines, points, s_points, coords, s_coords):
  """Initialise lines, points, and coordinates of an empty 3D animated graph
  both for GD and for SGD."""

  lines.set_data([], [])
  s_lines.set_data([], [])
  points.set_data([], [])
  s_points.set_data([], [])
  coords.set_text('')
  s_coords.set_text('')

  return lines, s_lines, points, s_points, coords, s_coords

def _update_graph3d(i,
  xs, ys, zs,
  s_xs, s_ys, s_zs,
  lines, s_lines,
  points, s_points,
  coords, s_coords
):
  """Update the animated chart's data simulating an iteration of (S)GD."""

  all_pts_x = [xs[j] for j in range(i)]
  all_pts_y = [ys[j] for j in range(i)]
  s_all_pts_x = [s_xs[j] for j in range(i)]
  s_all_pts_y = [s_ys[j] for j in range(i)]

  points.set_data(all_pts_x, all_pts_y)
  s_points.set_data(s_all_pts_x, s_all_pts_y)

  all_segs_x = _flatten_array(
    [list(np.linspace(xs[j], xs[j + 1], 50)) for j in range(i - 1)])
  all_segs_y = _flatten_array(
    [list(np.linspace(ys[j], ys[j + 1], 50)) for j in range(i - 1)])
  s_all_segs_x = _flatten_array(
    [list(np.linspace(s_xs[j], s_xs[j + 1], 50)) for j in range(i - 1)])
  s_all_segs_y = _flatten_array(
    [list(np.linspace(s_ys[j], s_ys[j + 1], 50)) for j in range(i - 1)])

  lines.set_data(all_segs_x, all_segs_y)
  s_lines.set_data(s_all_segs_x, s_all_segs_y)

  if(i > 0):
    coords.set_text("GD g({:.2f}, {:.2f}) = {:.2f}".format(
      xs[i - 1], ys[i - 1], zs[i - 1]))
    s_coords.set_text("SGD g({:.2f}, {:.2f}) = {:.2f}".format(
      s_xs[i - 1], s_ys[i - 1], s_zs[i -1]))

  return lines, s_lines, points, s_points, coords, s_coords

def animate_gd3d(**kwargs):
  """Produces an animation of the (S)GD algorithm applied to the example
  function `_sgd_g`.

  # Arguments

  start_x: float, first coordinate of the initial point
  start_y: float, second coordinate of the initial point
  step_size: float, learning rate of the GD algorithm
  num_iterations: int, number of GD algorithm iterations
  bounds_x: int[2], left and right bounds of the x axis in the animation
  bounds_y: int[2], left and right bounds of the y axis in the animation

  # Returns

  The animation object, which can be displayed in Jupyter.
  """

  start_x = kwargs.get('start_x', -0.5)
  start_y = kwargs.get('start_y', 4.5)
  step_size = kwargs.get('step_size', 0.1)
  num_iterations = kwargs.get('num_iterations', 100)
  bounds_x = kwargs.get('bounds_x', [-5.0, 6.0])
  bounds_y = kwargs.get('bounds_y', [-5.0, 6.0])

  xs, ys, zs = _gradient_points3d(start_x, start_y,
    step_size, num_iterations, False)
  s_xs, s_ys, s_zs = _gradient_points3d(start_x, start_y,
    step_size, num_iterations, True)

  fig, ax = plt.subplots(figsize = (10, 10))
  plt.close()

  x_axis_pts = np.arange(bounds_x[0], bounds_x[1], 0.05)
  y_axis_pts = np.arange(bounds_y[0], bounds_y[1], 0.05)
  f_pts_x, f_pts_y = np.meshgrid(x_axis_pts, y_axis_pts)
  f_pts_z = _sgd_g(f_pts_x, f_pts_y)

  ax.contour(f_pts_x, f_pts_y, f_pts_z,
    cmap = 'Greens', levels = np.linspace(0, np.amax(f_pts_z), 20))

  lines, = ax.plot([], [], 'r')
  s_lines, = ax.plot([], [], 'b')
  points, = ax.plot([], [], 'ro', markersize = 1)
  s_points, = ax.plot([], [], 'bo', markersize = 1)
  coords = ax.text(1, -4, '')
  s_coords = ax.text(1, -4.5, '')

  init_f = lambda: _init_graph3d(lines, s_lines,
    points, s_points,
    coords, s_coords)

  anim_f = lambda i: _update_graph3d(i,
    xs, ys, zs,
    s_xs, s_ys, s_zs,
    lines, s_lines,
    points, s_points,
    coords, s_coords)

  return anim.FuncAnimation(fig, anim_f, init_func = init_f,
    frames = num_iterations, interval = 50, blit = True, repeat = False)

def _gradient_points_mom(
  start_x,
  start_y,
  step_size = 0.1,
  num_iterations = 10,
  beta = 0.99
):
  """Gives all the points visited by the GD algorithm with momentum, applied to
  the example function `_sgd_g`.
  
  # Arguments

  start_x: float, first coordinate of the initial point
  start_y: float, second coordinate of the initial point
  step_size: float, learning rate of the algorithm
  num_iterations: int, number of iterations of gradient descent
  beta: float, momentum weight parameter

  # Returns

  A sextuple of lists. The first two contain all the points visited by GD (the x
  coordinate in the first list, and the y coordinate in the second), and the
  third contains the values of the function at these points. The last three
  are similar, but refer to GD+momentum.
  """

  xs = [start_x]
  mom_xs = [start_x]
  ys = [start_y]
  mom_ys = [start_y]
  zs = [_sgd_g(start_x, start_y)]
  mom_zs = [_sgd_g(start_x, start_y)]

  momentum_x = 0
  momentum_y = 0

  for i in range(num_iterations - 1):
    new_x = xs[i] - step_size * _sgd_g_diff_x(xs[i], ys[i])
    new_y = ys[i] - step_size * _sgd_g_diff_y(xs[i], ys[i])

    xs.append(new_x)
    ys.append(new_y)
    zs.append(_sgd_g(new_x, new_y))

    new_momentum_x = beta * momentum_x + _sgd_g_diff_x(mom_xs[i], mom_ys[i])
    new_momentum_y = beta * momentum_y + _sgd_g_diff_y(mom_xs[i], mom_ys[i])

    mom_new_x = mom_xs[i] - step_size * new_momentum_x
    mom_new_y = mom_ys[i] - step_size * new_momentum_y

    mom_xs.append(mom_new_x)
    mom_ys.append(mom_new_y)
    mom_zs.append(_sgd_g(mom_new_x, mom_new_y))

    momentum_x = new_momentum_x
    momentum_y = new_momentum_y

  return xs, ys, zs, mom_xs, mom_ys, mom_zs

def _init_graph_mom(lines, m_lines, points, m_points, coords, m_coords):
  """Initialise lines, points, and coordinates of an empty 3D animated graph
  both for GD and for GD+momentum."""

  lines.set_data([], [])
  m_lines.set_data([], [])
  points.set_data([], [])
  m_points.set_data([], [])
  coords.set_text('')
  m_coords.set_text('')

  return lines, m_lines, points, m_points, coords, m_coords

def _update_graph_mom(i,
  xs, ys, zs,
  m_xs, m_ys, m_zs,
  lines, m_lines,
  points, m_points,
  coords, m_coords
):
  """Update the animated chart's data simulating an iteration of
  GD(+momentum)."""

  all_pts_x = [xs[j] for j in range(i)]
  all_pts_y = [ys[j] for j in range(i)]
  m_all_pts_x = [m_xs[j] for j in range(i)]
  m_all_pts_y = [m_ys[j] for j in range(i)]

  points.set_data(all_pts_x, all_pts_y)
  m_points.set_data(m_all_pts_x, m_all_pts_y)

  all_segs_x = _flatten_array([list(np.linspace(xs[j], xs[j + 1], 50)) \
    for j in range(i - 1)])
  all_segs_y = _flatten_array([list(np.linspace(ys[j], ys[j + 1], 50)) \
    for j in range(i - 1)])
  m_all_segs_x = _flatten_array([list(np.linspace(m_xs[j], m_xs[j + 1], 50)) \
    for j in range(i - 1)])
  m_all_segs_y = _flatten_array([list(np.linspace(m_ys[j], m_ys[j + 1], 50)) \
    for j in range(i - 1)])

  lines.set_data(all_segs_x, all_segs_y)
  m_lines.set_data(m_all_segs_x, m_all_segs_y)

  if(i > 0):
    coords.set_text("GD g({:.2f}, {:.2f}) = {:.2f}".format(
      xs[i - 1], ys[i - 1], zs[i - 1]))
    m_coords.set_text("GD-momentum g({:.2f}, {:.2f}) = {:.2f}".format(
      m_xs[i - 1], m_ys[i - 1], m_zs[i -1]))

  return lines, m_lines, points, m_points, coords, m_coords

def animate_gd_mom(**kwargs):
  """Produces an animation of the GD(+momentum) algorithm applied to the
  example function `_sgd_g`.

  # Arguments

  start_x: float, first coordinate of the initial point
  start_y: float, second coordinate of the initial point
  step_size: float, learning rate of the GD algorithm
  num_iterations: int, number of GD algorithm iterations
  beta: float, momentum weight parameter
  bounds_x: int[2], left and right bounds of the x axis in the animation
  bounds_y: int[2], left and right bounds of the y axis in the animation

  # Returns

  The animation object, which can be displayed in Jupyter.
  """

  start_x = kwargs.get('start_x', -0.5)
  start_y = kwargs.get('start_y', 4.5)
  step_size = kwargs.get('step_size', 0.1)
  num_iterations = kwargs.get('num_iterations', 100)
  beta = kwargs.get('beta', 0.99)
  bounds_x = kwargs.get('bounds_x', [-5.0, 6.0])
  bounds_y = kwargs.get('bounds_y', [-5.0, 6.0])

  xs, ys, zs, m_xs, m_ys, m_zs = _gradient_points_mom(
    start_x, start_y, step_size, num_iterations, beta)

  fig, ax = plt.subplots(figsize = (10, 10))
  plt.close()

  x_axis_pts = np.arange(bounds_x[0], bounds_x[1], 0.05)
  y_axis_pts = np.arange(bounds_y[0], bounds_y[1], 0.05)
  f_pts_x, f_pts_y = np.meshgrid(x_axis_pts, y_axis_pts)
  f_pts_z = _sgd_g(f_pts_x, f_pts_y)

  ax.contour(f_pts_x, f_pts_y, f_pts_z,
    cmap = 'Greens', levels = np.linspace(0, np.amax(f_pts_z), 20))

  lines, = ax.plot([], [], 'r')
  m_lines, = ax.plot([], [], 'b')
  points, = ax.plot([], [], 'ro', markersize = 1)
  m_points, = ax.plot([], [], 'bo', markersize = 1)
  coords = ax.text(1, -4, '')
  m_coords = ax.text(1, -4.5, '')

  init_f = lambda: _init_graph_mom(lines, m_lines,
    points, m_points,
    coords, m_coords)

  anim_f = lambda i: _update_graph_mom(i,
    xs, ys, zs,
    m_xs, m_ys, m_zs,
    lines, m_lines,
    points, m_points,
    coords, m_coords)

  return anim.FuncAnimation(fig, anim_f,
    init_func = init_f, frames = num_iterations,
    interval = 50, blit = True, repeat = False)