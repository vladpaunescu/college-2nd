#!/usr/bin/python
from copy import deepcopy

import numpy as np

A = 10

class PSO:
  def __init__(self, func, b, S, omega, fp, fg):
    self.func = func
    self.b = b
    self.d = abs(b[1] - b[0])
    self.S, self.omega, self.fp, self. fg = S, omega, fp, fg
    self.x = self.generate(b[0], b[1])
    self.p = deepcopy(self.x) # best particle position
    self.v = self.generate(-self.d, self.d)
    self.g = self.get_best_swarm_p(self.func, self.x)
   # print "Best swarm g ", self.g

  def get_best_swarm_p(self, f, x):
    ys =map(f, x)
   # print "PLm", x
    return x[np.argmin(ys)]

  def run(self, n):
    for i in xrange(n):
      self.optimize(self.x, self.v, self.p, self.func)
    return self.g

  def optimize(self, x, v, p, f):
  #  print "g ", self.g
    for i in xrange(self.S):
      rp, rg = np.random.random(2)
      v[i] = self.omega * v[i] + self.fp * rp * (p[i] - x[i])\
                + self.fg * rg * (self.g - x[i])
      v[i] = np.array([constrain(el, -self.d, self.d) for el in v[i]])
      x[i] = x[i] + v[i]
      x[i] = np.array([constrain(el, self.b[0], self.b[1]) for el in x[i]])
      if f(x[i]) < f(p[i]):
        p[i] = np.array(x[i])
      if f(self.g) > f(x[i]):
        # print "Data ", i, " ", f(self.g), " ", f(x[i])
        self.g = np.array(x[i])
        # print "New global position ", self.g, " ", f(self.g)

  def generate(self, a, b):
    x = np.random.uniform(a, b, self.S)
    y = np.random.uniform(a, b, self.S)
    return np.array(zip(x, y))


def sphere(x):
  x = np.asarray(x)
  return np.dot(x, x)

def ackley(xs):
  x, y = xs[0], xs[1]
  return -20 * np.exp(0.2 * np.sqrt(0.5 * (x**2 + y**2))) \
         - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) \
         + np.e + 20

def rosenbrock(x):
  value = 0
  for i in range(len(x))[:-1]:
    value += 100 * (x[i+1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2

  return value

def rastrigin(x):
  n = len(x)
  value = A * n
  for el in x:
    value += (el ** 2 - A * np.cos(2 * np.pi * el))
  return value

def griewank(x):
  value = 1
  sum1 = 0
  prod1 = 1
  for i, el in enumerate(x):
    sum1 += (el ** 2)
    prod1 = prod1 * np.cos(el / float(np.sqrt(i + 1)))

  value += (sum1 / 4000.0)
  value -= prod1
  return value

def constrain(x, l, u):
  if x < l: return l
  if x > u: return u
  return x

if __name__ == "__main__":
  best_val = 1000
  best_g = None
  best_i = -1
  for i in range(10):
    pso = PSO(rosenbrock, (-5., 5.), 100, 0.3, -3, 3.6475)
    g = pso.run(100)
    if best_val > rosenbrock(g):
     best_val = rosenbrock(g)
     best_g = g
     best_i = i

  print "Best g ", best_g, " ", best_val, " ", best_i

  # pso2 = PSO(rosenbrock, (-5, 5.), 100, 0.3, -3, 3.6475)
  # pso2.run(500)

  #pso3 = PSO(rastrigin, (-5.12, 5.12), 100, 0.3, -3, 3.6475)
  #pso3.run(500)

  #pso4 = PSO(griewank, (-600, 600), 100, 0.39, 2.05, 2.05)
  #pso4.run(5000)
  #print griewank([-600, -217])

