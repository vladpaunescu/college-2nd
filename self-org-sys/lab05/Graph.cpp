/*
 * Graph.cpp
 *
 *  Created on: Dec 6, 2014
 *      Author: vlad
 */

#include "Graph.h"

#include <algorithm>
#include <cstdio>

#include "Utils.h"

extern float LO;
extern float HI;

Graph::Graph(int _n) : n(_n) {
	this->generateGraph();

}

Graph::~Graph() {

}

void Graph::allocate(int n) {
	costs = vector<vector<float> >(n);
	std::for_each(
			costs.begin(),
			costs.end() ,
			[&n] (vector<float>& el) {
		el = vector<float>(n);
	} );
}

void Graph::generateGraph() {
	this->allocate(n);

	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			costs[i][j] = costs[j][i] = Utils::getRandom(LO, HI);
		}
	}
}

void Graph::printGraph() {

	std::for_each(
			costs.begin(),
			costs.end(),
			[] (vector<float>& row) {
		for (auto& el : row) {
			printf("%8.5f ", el);
		}
		putchar('\n');
	} );
}


