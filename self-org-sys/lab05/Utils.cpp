
#include "Utils.h"

#include <cstdio>
#include <cstdlib>


Utils::Utils() {
	// TODO Auto-generated constructor stub

}

float Utils::getRandom(const float low, const float high) {
	return low
			+ static_cast<float>(rand())
			/ (static_cast<float>(RAND_MAX / (high - low)));

}

void Utils::printPath(float length, vector<int>& path) {
	printf("Path length %8.5f ", length);
	for (auto& el : path) {
		printf("%d ", el);
	}
	putchar('\n');
}


Utils::~Utils() {
	// TODO Auto-generated destructor stub
}

