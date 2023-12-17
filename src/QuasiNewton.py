import numpy as np
import matplotlib.pyplot as plt
import math 

def zoom_line_search(a_low, a_high, f, phi, D_phi, x_k, p_k, c_1, c_2):
  '''Implementation of the zoom subroutine for Wolfe Condition Line Search'''
  phi_zero = phi(0, f, x_k, p_k) # value of phi at zero
  D_phi_zero = D_phi(0, x_k, p_k) # value of derivative of phi at zero

  while True:
    a_j = (a_low + a_high) / 2

    phi_a = phi(a_j, f, x_k, p_k) # value of phi at curr alpha
    phi_a_low = phi(a_low, f, x_k, p_k) # value of phi at low alpha
    D_phi_a = D_phi(a_j, x_k, p_k) # value of derivative of phi at curr alpha

    if (phi_a > (phi_zero + c_1 * a_j * D_phi_zero)) or (phi_a >= phi_a_low):
      a_high = a_j
    else:
      if (abs(D_phi_a) <= c_2 * abs(D_phi_zero)):
        return a_j

      if (D_phi_a * (a_high - a_low)) >= 0:
        a_high = a_low

      a_low = a_j

def line_search_wolfe_cond(f, phi, D_phi, x_k, p_k, c_1=0.001, c_2=0.9):
  '''Implementation of Wolfe Condition Line Search'''
  a_prev = 0.0
  a_max = 1 # choose a_max > 0
  a_i = (a_max - a_prev) / 2 # note: returns float
  i = 1

  phi_zero = phi(0, f, x_k, p_k) # value of phi at zero
  D_phi_zero = D_phi(0, x_k, p_k) # value of derivative of phi at zero

  while True:
    phi_a = phi(a_i, f, x_k, p_k) # value of phi at curr alpha
    D_phi_a = D_phi(a_i, x_k, p_k) # value of derivative of phi at curr alpha
    phi_a_prev = phi(a_prev, f, x_k, p_k) # value of phi at prev alpha

    if (phi_a > (phi_zero + c_1 * a_i * D_phi_zero)) or ((a_i >= phi_a_prev) and i > 1):
      return zoom_line_search(a_prev, a_i, f, phi, D_phi, x_k, p_k, c_1, c_2)

    if (abs(D_phi_a) <= c_2 * abs(D_phi_zero)):
      return a_i

    if (D_phi_a >= 0):
      return zoom_line_search(a_prev, a_i, f, phi, D_phi, x_k, p_k, c_1, c_2)

    a_prev = a_i
    a_i = (a_i + a_max) / 2
    i += 1

debug = False

def DEBUG(s):
  if debug: print(s)

def sin_squared(x):
  return pow(math.sin(x), 2)

def cos_squared(x):
  return pow(math.cos(x), 2)

def problem5_phi(a, f, x, p):
  '''One-dimensional function for which we are approximating a minumum'''
  return f(x + a * p)

def problem5_D_phi(a, x, p):
  '''First derivative of the function problem5_phi when f = problem5_f (with respect to a)'''
  component_0 = x[0] + a * p[0]
  component_1 = x[1] + a * p[1]
  sin = math.sin
  cos = math.cos
  d_chain_outer = 2 * (pow(sin(component_0), 2) + pow(cos(component_1), 2))
  d_sin_squared = 2 * p[0] * sin(component_0) * cos(component_0)
  d_cos_squared = -2 * p[1] * cos(component_1) * sin(component_1)
  d_chain_inner = d_sin_squared + d_cos_squared
  return d_chain_outer * d_chain_inner

def problem5_f(x):
  '''Implementation of the function provided in the prompt'''
  return pow(sin_squared(x[0]) + cos_squared(x[1]), 2)

def problem5_gradient_f(x):
  '''Gradient of problem5_f'''
  sin = math.sin
  cos = math.cos
  d_x0 = 4 * sin(x[0]) * cos(x[0]) * (sin_squared(x[0]) + cos_squared(x[1]))
  d_x1 = -4 * sin(x[1]) * cos(x[1]) * (sin_squared(x[0]) + cos_squared(x[1]))
  return np.array([d_x0, d_x1])

def bfgs_direct_inverse_update(s_k, y_k, h_curr):
  '''Directly update the hessian approximation using BFGS Updates'''
  # print(f"y_k: {y_k}")
  # print(f"s_k: {s_k}")
  # print("Calculating direct inverse")
  p_k = 1 / (y_k @ s_k)
  first = np.eye(2) - p_k * s_k @ y_k.T
  # print(f"First: {first}")
  second = h_curr
  DEBUG(f"Second: {second}")
  third = np.eye(2) - p_k * y_k.T @ s_k
  DEBUG(f"Third: {third}")
  fourth = p_k * s_k @ s_k.T
  DEBUG(f"Fourth: {fourth}")
  DEBUG("DONE calculating direct inverse")
  return first @ second @ third + fourth

def problem5_solution(x0, h0, f, phi, D_phi, grad):
  '''Solution for problem 5, returns a list of the values of f for each x in the iteration'''
  x_track = []
  x_curr = x0
  x_prev = x0 * 1
  h_curr = h0
  g_curr = grad(x_curr)
  g_prev = g_curr * 1
  i = 0
  for i in range(100):
  # while (np.linalg.norm(g_curr) > 0.000001):
    DEBUG(f"Iteration {i}")
    # pick a direction pk
    p_curr = -1 * h_curr @ g_curr
    DEBUG(f"pcurr: {p_curr}")
    # solve the one-dimensional problem using line search
    a = line_search_wolfe_cond(f, phi, D_phi, x_curr, p_curr)
    DEBUG(f"alpha: {a}")
    # update position vector x
    x_curr = x_prev + a * p_curr
    DEBUG(f"x_curr: {x_curr}")
    # update the gradient
    g_curr = grad(x_curr)
    DEBUG(f"g_curr: {g_curr}")
    # calculate new estimate of hessian matrix
    s_k = x_curr - x_prev
    y_k = g_curr - g_prev
    DEBUG(f"sk: {s_k} yk = {y_k}")
    h_curr = bfgs_direct_inverse_update(s_k, y_k, h_curr)
    DEBUG(f"h_curr: {h_curr}")
    # update state
    g_prev = g_curr
    x_prev = x_curr
    # objective output of the i-th iterate
    x_track.append(f(x_curr))
    DEBUG(f"Iteration {i} done")
    i += 1
  print(f"x_star: {x_curr}")
  return x_track

np.random.seed(44565) # do not change
x0 = np.random.uniform(-2.0, 2.0, 2) # do not change
H0 = np.eye(2) # do not change

print(f"x_0: {x0}")

sol = problem5_solution(x0, H0, problem5_f, problem5_phi, problem5_D_phi, problem5_gradient_f)
plt.plot(sol)