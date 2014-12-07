/*
 * Tsp.cpp
 *
 *  Created on: Dec 7, 2014
 *      Author: vlad
 */

#include "Tsp.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <utility>

#include "Utils.h"



Tsp::Tsp(Graph& _g) : g(_g)  {
	bestPathLength =  numeric_limits<float>::max();
}

void Tsp::findOptimalPath() {
	vector<int> path;
	set<int> visited;
	//printf("pls\n");
	findOptimalPath(path, visited);
}

void Tsp::findOptimalPath(vector<int>& path , set<int>& visited) {
	Utils::printPath(bestPathLength, path);
	//printf("pls\n");
	if (this->getPathLength(path) > this->bestPathLength) {
		//printf("best %f\n", bestPathLength);
		return;
	}
	//printf("sdasda\n");
	if (completePath(path)) {
		path.push_back(path[0]);
		if (getPathLength(path) < bestPathLength) {
			printf("complete");
			bestPath = path;
			bestPathLength = getPathLength(path);
		}
		path.pop_back();
		return;
	}
	//Utils::printPath(bestPathLength, path);

	for (int i = 0; i < g.n; ++i) {
	//	printf("%d\n" , i);
		if (visited.find(i) == visited.end()) {
			visited.insert(i);
			path.push_back(i);
			findOptimalPath(path, visited);
			visited.erase(i);
			path.pop_back();
		}
	}
}

float Tsp::getPathLength(vector<int>& path) {
	float sum = 0;

	//printf("path size %d\n", path.size());

	for (std::vector<int>::size_type i = 0; i != path.size(); i++) {
	    /* std::cout << someVector[i]; ... */
		sum += g.costs[path[i]][path[i+1]];
	}

	return sum;
}

bool Tsp::completePath(vector<int>& path) {
	return path.size() == g.n;
}

Tsp::~Tsp() {
	// TODO Auto-generated destructor stub
}

