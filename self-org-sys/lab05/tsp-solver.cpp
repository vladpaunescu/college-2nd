#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <vector>

#include "Graph.h"
#include "Tsp.h"
#include "Utils.h"

using namespace std;

const int INF = 10000;


float LO = 1.0;
float HI = 100.0;

int n;
vector<vector<float> > costs;
float best_cost = numeric_limits<float>::max();
int *best_path;


float get_cost(int sol[], int k){
	int sum = 0;
	for (int i = 0; i <= k; ++i) {
		sum += costs[sol[i]][sol[i+1]];
	}
	return sum;
}

//class Tsp {
//
//	public Tsp(int n, Graph graph) {
//
//	}
//
//};

void print_sol(int sol[], const float& cost) {
	printf("Cost %f\n", cost);
	for(int i = 0; i <= n; ++i) {
		printf("%d ", sol[i] + 1);
	}
	putchar('\n');
}

void print_sol_partial(int sol[], int k) {
	for(int i = 0; i < k; ++i) {
		printf("%d ", sol[i] + 1);
	}
	putchar('\n');
}

//void tsp_solve(vector<int> sol, int k) {
//	if (k == n + 1) {
//		float sum = get_cost(sol, k);
//		if (sum < best_cost) {
//			best_cost = sum;
//			std::copy(sol , sol + n, best_path);
//			print_sol(best_path, sum);
//		}
//		return;
//	}
//
//	for (int i = 0; i < n; ++i) {
//		if (k == 0) {
//			sol[k] = i;
//			visited[i] = 1;
//			tsp_solve(sol, visited, k+1);
//			visited[i] = 0;
//		} else if (0 < k && k < n && !visited[i] && costs[sol[k-1]][i] != INF) {
//			sol[k] = i;
//			visited[i] = 1;
//			tsp_solve(sol, visited, k+1);
//			visited[i] = 0;
//		}
//	}
//	if (k == n && costs[sol[k-1]][sol[0]] != INF) {
//		sol[k] = sol[0];
//		tsp_solve(sol, visited, k+1);
//	}
//}

int main() {
	freopen("data.in", "r", stdin);
	scanf("%d", &n);
	printf("%d\n", n);

	srand (static_cast <unsigned> (time(0)));

	Graph g(5);
	g.printGraph();

	Tsp tsp(g);
	printf("Finding optimal path...\n");
	tsp.findOptimalPath();
	printf("Optimal path length...\n");
	Utils::printPath(tsp.bestPathLength, tsp.bestPath);


//
//	for (int i = 0; i < n; ++i) {
//		costs.push_back(vector<float>());
//		for (int j = 0; j < n; ++j) {
//			float aux;
//			scanf("%f", &aux);
//			costs[i].push_back(aux);
//			printf("%f ", costs[i][j]);
//		}
//		putchar('\n');
//	}
//	vector<int> solution;
//	best_path = new int[n + 1];
//
//	for (int i = 0; i <= n; ++i) {
//		printf("%d ", visited[i]);
//	}
//	tsp_solve(solution, 0);

	return 0;
}
