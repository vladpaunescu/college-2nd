import random
import sys

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



def computeAttractiveness(eta, tetha, alpha, beta, n):
    l = len(eta)
    a = []
    for i in range(n):
        a.append(n * [0])

    for i in range(l):
        s = 0.0
        for j in range(l):
            if i == j:
                continue
            a[i][j] = pow(tetha[i][j], alpha) * pow(eta[i][j], beta)
            s += a[i][j]
        for j in range(l):
            if i == j:
                continue
            a[i][j] /= s

    return a

def evaporate(tetha, rho):
    l = len(tetha)

    for i in range(l):
        for j in range(l):
            tetha[i][j] = (1 - rho) * tetha[i][j]


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



#m = nbr of ants
def TSPants(alpha, beta, rho, Q, G, m, n, NCmax, optimalCost, tetha0 = 0.1):
    l = len(G)


    #print '--------------'
    tetha = []
    for i in range(l):        
        tetha.append(l * [tetha0])
    #printMatrix(tetha)

    #print '----------------'

    eta = []
    for i in range(l):        
        eta.append(l * [0])

    for i in range(0, l):
        for j in range(i + 1, l):
            eta[i][j] = 1.0 / G[i][j]
            eta[j][i] = 1.0 / G[j][i]
    #printMatrix(eta)
    #print '----------------'

    a = computeAttractiveness(eta, tetha, alpha, beta, n)

    bestSol = []
    bestSolCost = sys.maxint

    for it in range(NCmax):
        #print 'iter = ', it
        ants = []
        for i in range(m):
            startCity = random.randint(0, l - 1)
            ants.append(Ant(i, startCity, G))

        
        #after each ant made a tour
        for step in range(l - 1):
            for ant in ants:
                ant.run(a)

        for ant in ants:
            cost = ant.solutionLength()
            if cost < bestSolCost:
                bestSolCost = cost
                bestSol = ant.sol[:]
                #print '>> ', cost, ant.sol    

        for step in range(l - 1, 0, -1):
            #print 'step: ', step
            for ant in ants:
                tetha = ant.updateTetha(tetha, step, Q)
                a = computeAttractiveness(eta, tetha, alpha, beta, n)
#                print '~_~_~_~_~_~_~_~'
#                printMatrix(tetha)

        del ants[:]

        evaporate(tetha, rho)
    
    #print '------------------------\n', 'approx solution'
    #printMatrix(a)
    #print '------------------------'
    print 'solution: ', bestSol
    #print 'cost: ', bestSolCost
    print 'error: ', bestSolCost - optimalCost

    return [bestSolCost, bestSol]


def experiment(G, Q, m, NCmax, alpha, beta, rho, n, optimalCost):
    bestCost = sys.maxint
    bestSol = []
    for ex in range(10):
        cost, sol = TSPants(alpha, beta, rho, Q, G, m, NCmax, optimalCost)
        if cost < bestCost:
            bestCost = cost
            bestSol = sol[:]
        #print cost, sol
    error = bestCost - optimalCost
    return error


class Ant:
    def __init__(self, antId, startCity, G):
        self.antId = antId
        self.startCity = startCity
        self.G = G
        self.sol = [self.startCity]

    def updateTetha(self, tetha, i, Q):
        l = 0
        for j in range(len(self.G) - 1):
            #print self.sol[i], self.sol[i + 1], len(self.sol)
            l += self.G[self.sol[j]][self.sol[j + 1]]

        
        #tetha[self.sol[i]][self.sol[i - 1]] += 1.0 / l
        tetha[self.sol[i - 1]][self.sol[i]] += Q * 1.0 / l

        return tetha

    def solutionLength(self):
        length = 0
        l = len(self.sol)
        for i in range(l - 1):
            length += self.G[self.sol[i]][self.sol[i + 1]]
        return length
                 
    #compute p for ant ant
    def computeP(self, a):
        l = len(self.G)
        p = []
        for i in range(l):
            p.append(l * [0])

        i = self.sol[-1]
        s = 0
        for j in range(l):
            if i == j or j in self.sol:
                continue
            s += a[i][j]

        for j in range(l):
            if i == j or j in self.sol:
                p[i][j] = 0
            else: 
                p[i][j] = a[i][j] / s

        return p


    #choose node by probability
    def chooseNode(self, p):
        l = len(p)
        prob = random.random()
    
        i = self.sol[-1]

        probs = [pr for pr in p[i] if pr > 0]

        indices = [ind for ind, x in enumerate(p[i]) if x > 0]

        for j in range(1, len(probs)):
            probs[j] = probs[j - 1] + probs[j]

        if len(probs) == 0:
            return None

        for j in range(len(probs)):
            if prob <= probs[j]:
                ind = indices[j]
                return ind
        return None
                
            

    def run(self, a):
        #print '~~~~~~~~~~~~~~~~~~~~'
        n = len(self.sol)

        p = self.computeP(a)
        nextNode = self.chooseNode(p)
        self.sol.append(nextNode)

        #print self.antId, ' sol len: ', len(self.sol), 'sol: ', self.sol
        



if __name__ == "__main__":

    n = 4

    #graph = generateGraph(n)
    #printMatrix(graph)


    '''ants = []
    for i in range(len(graph)):
        ants[i] = Ant(i, i)


    TSPants(graph, 0.1, 3, 0.5, 0.5)'''


    #tsp = TSP(n, graph)
    #tsp.findTSPPath()
    #print tsp.bestPath
    #print tsp.bestPathLength

    #optimal solution
    """tsp = TSP(n, graph)
    tsp.findTSPPath()
    optimalCost = tsp.bestPathLength"""

    NCmax = 30
    m = 15
    Q = 0.7

    '''alpha = 1
    beta = 5
    rho = 0.1'''
    
    '''TSPants(alpha, beta, rho, Q, graph, m, NCmax, optimalCost)

    print '-------------'
    printMatrix(graph)

    
    '''

    alphas = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    betas =  [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2, 0.1]
    rhos =   [0.1, 0.01, 0.1, 0.2, 0.1, 0.01, 0.05, 0.1, 0.4, 0.3]
    dims = [9, 10, 11]


    bestErrs = len(dims) * [0]
    bestExps = len(dims) * [0]

    # print "{0}{1}{2}".format(len(alphas), len(betas), len(rhos))
    for i in range(len(dims)):
        n = dims[i]

        G = generateGraph(n)
        printMatrix(G)

        minErr = sys.maxint
        bestExp = -1
        print 'graph dim = ', n
        tsp = TSP(n, G)
        tsp.findTSPPath()
        optimalCost = tsp.bestPathLength
        print 'optimal cost: ', optimalCost, 'sol: ', tsp.bestPath
        for ex in range(len(alphas)):
            error = experiment(G, Q, m, NCmax, alphas[ex], betas[ex], rhos[ex], n, optimalCost)
            if error < minErr:
                minErr = error
                bestExp = ex
            print ex, ':', minErr, '(', alphas[ex], betas[ex], rhos[ex], ')'
    
        bestErrs[i] = minErr
        bestExps[i] = bestExp

    for i in range(len(dims)):
        print dims[i], bestErrs[i], '(', alphas[bestExps[i]], betas[bestExps[i]], rhos[bestExps[i]], ')'

    #print '-------------\noptimal solution'
    #print tsp.bestPath
    #print tsp.bestPathLength



