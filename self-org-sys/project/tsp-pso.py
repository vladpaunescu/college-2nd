from duplicity.pexpect import searcher_re
import random
import sys
import numpy as np
import perm_util as pu

def generateGraph(nbrNodes):
    graph = []
    for i in range(nbrNodes):
        graph.append(nbrNodes * [0])

    for i in range(nbrNodes):
        for j in range(i + 1, nbrNodes):
            graph[i][j] = random.random() * 100 + 10
            graph[j][i] = graph[i][j]
    return graph

def printMatrix(m):
    print "\n".join(' '.join(str(x) for x in l) for l in m)




class TSP:
    def __init__(self, n, G):
        self.bestPathLength = sys.maxint
        self.bestPath = []
        self.n = n
        self.G = G

    def completePath(self, pth):
        pthSet = set(pth)
        if len(pthSet) == self.n:
            return True
        return False


    def pathLength(self, pth):
        length = 0
        l = len(pth)
        for i in range(l - 1):
            length += self.G[pth[i]][pth[i + 1]]
        return length


    def findTSPPath(self, pth = []):
        if self.pathLength(pth) > self.bestPathLength:
            return
        if self.completePath(pth) and self.pathLength(pth) < self.bestPathLength:
            self.bestPathLength = self.pathLength(pth)
            self.bestPath = pth[:]
            return

        for i in range(self.n):
            if not i in pth:
                pth.append(i)
                self.findTSPPath(pth)
                del pth[-1]

class Particle:

    def __init__(self, G, n, v_length):
        self.G, self.n = G, n
        self.v_length = v_length
        self.perm, self.vs = self.generate()
        self.pbest = (self.get_length(), self.perm)
        print "Particle\n\t{}\n\t{}".format(self.perm, self.vs)


    def generate(self):
        perm = np.random.permutation(n).tolist()
        vs = []
        for _ in xrange(self.v_length):
            v = tuple(np.random.randint(n, size=2))
            while v in vs:
                v = tuple(np.random.randint(n, size=2))
            vs.append(v)
        return perm, vs

    def update_pbest(self):
        l = self.get_length()
        if l < self.pbest[0]:
            self.pbest = (l, self.perm)

    """
    The probability that all swap operators in swap sequence
    (pbest - x(t-1)) are included in the updated velocity is alpha.

    The probability that all swap operators in swap sequence
    (gbest - x(t-1)) are included in the updated velocity is beta.
    """
    def update_speed(self, alpha, beta, gama, gbest):
        pss = pu.subtract(self.pbest[1], self.perm)
        gss = pu.subtract(gbest[1], self.perm)
        self.vs = pu.prob_concat((), self.vs, gama)
        self.vs = pu.prob_concat(self.vs, pss, alpha)
        self.vs = pu.prob_concat(self.vs, gss, beta)


    def update_position(self):
        self.perm = pu.apply_swap_seq(self.perm, self.vs)


    def get_length(self):
        length = 0
        l = len(self.perm)
        for i in range(l - 1):
            length += self.G[self.perm[i]][self.perm[i + 1]]
        return length


class PSO:

    def __init__(self, G, n, part_count, v_length, steps, alpha, beta):
        self.G = G
        self.n, self.v_length = n, v_length
        self.part_count = part_count
        self.gbest = (float("inf"), [])
        self.steps, self.alpha, self.beta = steps, alpha, beta
        self.gama = 1.0

        self.particles = [Particle(G, n, v_length) for _ in xrange(part_count)]
        for particle in self.particles:
            if particle.pbest[0] < self.gbest:
                self.gbest = particle.pbest
        print "Gbest {}".format(self.gbest)


    def run(self):
        for i in xrange(self.steps):
          #  print "Epoch {}".format(i)
            self.update_gbest()
            for particle in self.particles:
                particle.update_pbest()
                particle.update_speed(self.alpha, self.beta, self.gama, self.gbest)
                particle.update_position()
                self.alpha -= 0.002
                self.beta += 0.002
                self.gama -=0.0005

    def update_gbest(self):
        for particle in self.particles:
            l = particle.get_length()
            if l < self.gbest[0]:
                self.gbest = (l, particle.perm)

if __name__ == "__main__":

    dims = [9, 10, 11]
    part_count = [500, 100, 100]

    for i in range(len(dims)):
        n = dims[i]
        G = generateGraph(n)
        tsp = TSP(n, G)
        tsp.findTSPPath()
        optimalCost = tsp.bestPathLength
        print 'optimal cost: ', optimalCost, 'sol: ', tsp.bestPath

        pso = PSO(G, n, part_count[i], 2, 1000, 0.8, 0.2)
        pso.run()
        print pso.gbest
        break




