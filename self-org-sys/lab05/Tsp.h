/*
 * Tsp.h
 *
 *  Created on: Dec 7, 2014
 *      Author: vlad
 */

#ifndef LAB05_TSP_H_
#define LAB05_TSP_H_

#include <set>
#include <vector>

#include "Graph.h"

using namespace std;

class Tsp {
public:
	float bestPathLength;
	vector<int> bestPath;
	Graph& g;

	Tsp(Graph& g);

	void findOptimalPath();
	void findOptimalPath(vector<int>& path , set<int>& visited);
	float getPathLength(vector<int>& path);
	bool completePath(vector<int>& path);

	virtual ~Tsp();
};

#endif /* LAB05_TSP_H_ */
