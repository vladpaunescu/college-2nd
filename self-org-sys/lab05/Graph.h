/*
 * Graph.h
 *
 *  Created on: Dec 6, 2014
 *      Author: vlad
 */

#ifndef LAB05_GRAPH_H_
#define LAB05_GRAPH_H_

#include <vector>


using namespace std;

class Graph {
public:
	int n;
	vector<vector<float> > costs;

	Graph(int n);

	void generateGraph();
	void allocate(int n);
	void free_();
	void printGraph();

	virtual ~Graph();
};

#endif /* LAB05_GRAPH_H_ */
