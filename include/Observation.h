#include <vector>
#include <iostream>
using namespace std;

#ifndef OBSERVATION_H
#define OBSERVATION_H

class Observation{

public:
    vector<double> values;

    size_t size();
};

#endif
