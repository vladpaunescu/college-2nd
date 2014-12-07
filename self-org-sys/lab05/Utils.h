/*
 * utils.h
 *
 *  Created on: Dec 7, 2014
 *      Author: vlad
 */

#ifndef LAB05_UTILS_H_
#define LAB05_UTILS_H_

#include <vector>


extern float LO;
extern float HI;

using namespace std;

class Utils {
public:
	Utils();
	static float getRandom(const float low, const float high);
	static void printPath(float length, vector<int>& path);

	virtual ~Utils();

};

#endif /* LAB05_UTILS_H_ */
