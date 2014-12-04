#include <cstdio>
#include <vector>
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <limits>
#include <algorithm>    // std::copy
#include <iterator>
#include <set>

using namespace std;

const int INF = 1000;

int n;
vector<vector<float> > costs;
float best_cost = numeric_limits<float>::max();
int *best_path;

float get_cost(int sol[]){
	int sum = 0;
	for (int i = 0; i <= n; ++i) {
		sum += costs[sol[i]][sol[i+1]];
	}
	return sum;
}

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

void tsp_solve(int sol[], int *visited, int k) {
	if (k == n + 1) {
		float sum = get_cost(sol);
		if (sum < best_cost) {
			best_cost = sum;
			std::copy(sol , sol + n, best_path);
			print_sol(best_path, sum);
		}
		return;
	}

	for (int i = 0; i < n; ++i) {
		if (k == 0) {
			//	print_sol_partial(sol, k);
			sol[k] = i;
			visited[i] = 1;
			tsp_solve(sol, visited, k+1);
			visited[i] = 0;
		} else if (0 < k && k < n && !visited[i] && costs[sol[k-1]][i] != INF) {
			//print_sol_partial(sol, k);
			sol[k] = i;
			visited[i] = 1;
			tsp_solve(sol, visited, k+1);
			visited[i] = 0;
		}
	}
	if (k == n && costs[sol[k-1]][sol[0]] != INF) {
		//print_sol_partial(sol, k);
		sol[k] = sol[0];
		tsp_solve(sol, visited, k+1);
	}
}

int main() {
	freopen("data.in", "r", stdin);
	scanf("%d", &n);
	printf("%d\n", n);
	for (int i = 0; i < n; ++i) {
		costs.push_back(vector<float>());
		for (int j = 0; j < n; ++j) {
			float aux;
			scanf("%f", &aux);
			costs[i].push_back(aux);
			printf("%f ", costs[i][j]);
		}
		putchar('\n');
	}
	int *sol = new int[n + 1];
	best_path = new int[n + 1];
	int *visited = new int[n];
	for (int i = 0; i <= n; ++i) {
		printf("%d ", visited[i]);
	}
	tsp_solve(sol, visited, 0);

	return 0;
}
