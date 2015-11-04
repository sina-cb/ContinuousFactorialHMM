#include <vector>
#include <iostream>
using namespace std;

#ifndef SAMPLE_H
#define SAMPLE_H

class Sample{

public:
    vector<double> values;
    double p;

    size_t size();
};

#endif
