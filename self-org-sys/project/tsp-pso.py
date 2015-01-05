from duplicity.pexpect import searcher_re
import random
import sys
import numpy as np
import perm_util as pu
import matplotlib.pyplot as plt

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

    # def __init__(self, G, n, v_length):
    #     self.G, self.n = G, n
    #     self.v_length = v_length
    #     self.perm, self.vs = self.generate()
    #     self.pbest = (self.get_length(), self.perm)
    #     print "Particle\n\t{}\n\t{}".format(self.perm, self.vs)


    def __init__(self, G, n, v_length, bestPath = None):
        self.G, self.n = G, n
        self.v_length = v_length
        self.perm, self.vs = self.generate()
        # if bestPath is not None:
        #     self.perm = bestPath
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

    def __init__(self, tsp, G, n, part_count, v_length, steps, alpha, beta):
        self.G = G
        self.n, self.v_length = n, v_length
        self.part_count = part_count
        self.gbest = (float("inf"), [])
        self.steps, self.alpha, self.beta = steps, alpha, beta
        self.gama = 1.0
        self.optimum = tsp

        self.particles = [Particle(G, n, v_length) for _ in xrange(part_count)]
        self.particles[0] = Particle(G, n, v_length, tsp.bestPath)
        for particle in self.particles:
            if particle.pbest[0] < self.gbest:
                self.gbest = particle.pbest
        print "Gbest {}".format(self.gbest)

        self.errors = []
        self.gbest_trend = []
        self.alphas, self.betas, self.gamas = [], [], []



    def run(self):
        no_improvement_count = 0
        for i in xrange(self.steps):
            #  print "Epoch {}".format(i)
            self.alphas.append(self.alpha)
            self.betas.append(self.beta)
            self.gamas.append(self.gama)

            if not self.update_gbest():
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            for particle in self.particles:
                particle.update_pbest()
                particle.update_speed(self.alpha, self.beta, self.gama, self.gbest)
                particle.update_position()

            if no_improvement_count < 20:
                self.alpha -= 0.002
                self.beta += 0.002
                self.gama -= 0.0005
            else:
                self.alpha += 0.002
                self.beta -= 0.002
                self.gama +=0.0005


    def update_gbest(self):
        improve = False
        for particle in self.particles:
            l = particle.get_length()
            if l < self.gbest[0]:
                self.gbest = (l, particle.perm)
                improve = True

        self.gbest_trend.append(self.gbest)
        self.errors.append(self.gbest[0] - self.optimum.bestPathLength)
        return improve


def plot_errors(pso, directory):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    fig1.suptitle("Evolutia erorii")
    ax1.plot(range(len(pso.gbest_trend)), [gbest[0] for gbest in pso.gbest_trend], label="Global Best trend")
    ax1.plot(range(len(pso.errors)), pso.errors, label="Error (absolute)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    ax1.autoscale()
   # plt.savefig("{0}/plot_{1}_absolute".format(directory, thread_id))
   # plt.clf()
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    fig2.suptitle("Evolutia alpha beta gama")
    ax2.plot(range(len(pso.alphas)), pso.alphas, label="Alpha trend")
    ax2.plot(range(len(pso.betas)), pso.betas, label="Beta trend")
    ax2.plot(range(len(pso.betas)), pso.gamas, label="Gama trend")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
    ax2.autoscale()
    fig1.savefig("{0}/plot_error".format(directory))
    fig2.savefig("{0}/plot_alphas".format(directory))
    plt.show()





if __name__ == "__main__":

    dims = [9, 10, 11]
    part_count = [1000, 100, 100]

    for i in range(len(dims)):
        n = dims[i]
        G = generateGraph(n)
        printMatrix(G)
        tsp = TSP(n, G)
        tsp.findTSPPath()
        optimalCost = tsp.bestPathLength
        print 'optimal cost: ', optimalCost, 'sol: ', tsp.bestPath

        pso = PSO(tsp, G, n, part_count[i], 3, 100, 0.8, 0.2)
        pso.run()
        print pso.gbest
        print pso.gamas
        print pso.alphas
        print pso.betas
        plot_errors(pso, "/home/vlad/git/college-2nd/self-org-sys/project/plots")
        break




