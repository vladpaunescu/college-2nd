from duplicity.pexpect import searcher_re
import random
import sys
import numpy as np

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

    def __init__(self, n, v_length):
        self.n = n
        self.v_length = v_length
        self.perm, self.vs = self.generate()
        print "Particle\n\t{}\n\t{}".format(self.perm, self.vs)

    def generate(self):
        perm = np.random.permutation(n)
        vs = []
        for _ in xrange(self.v_length):
            v = tuple(np.random.randint(n, size=2))
            while v in vs:
                v = tuple(np.random.randint(n, size=2))
            vs.append(v)
        return perm, vs

class PSO:

    def __init__(self, n, part_count, v_length):
        self.n, self.v_length = n, v_length
        self.part_count = part_count
        self.particles = [Particle(n, v_length) for _ in xrange(part_count)]

if __name__ == "__main__":

    dims = [9, 10, 11]
    part_count = [100, 100, 100]

    for i in range(len(dims)):
        n = dims[i]
        G = generateGraph(n)
        p = PSO(n, part_count[i], 2)
        break




