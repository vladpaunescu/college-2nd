#!/usr/bin/python
from copy import deepcopy

import numpy as np

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
    print "Best swarm g ", self.g

  def get_best_swarm_p(self, f, x):
    ys =map(f, x)
    print ys
    return x[np.argmin(ys)]

  def run(self, n):
    for i in xrange(n):
      self.optimize(self.x, self.v, self.p, self.func)

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
        # print "New p, i ", p[i], " ", i
        p[i] = x[i]
      if f(self.g) > f(x[i]):
        self.g = x[i]
        print "New global position ", self.g, " ", f(self.g)

  def generate(self, a, b):
    x = np.random.uniform(a, b, self.S)
    y = np.random.uniform(a, b, self.S)
    return np.array(zip(x, y))


def sphere(x):
  x = np.asarray(x)
  return np.dot(x, x)

def constrain(x, l, u):
  if x < l: return l
  if x > u: return u
  return x

if __name__ == "__main__":
  pso = PSO(sphere, (-5.12, 5.12), 1000, -0.6031, -0.6485, 2.6475)
  pso.run(100)

