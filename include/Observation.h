#include <vector>
#include <iostream>
#include "Sample.h"
using namespace std;

#ifndef OBSERVATION_H
#define OBSERVATION_H

class Observation{

public:
    vector<double> values;
    Sample combine(Sample second);
    Observation convert(Sample sample);

    size_t size();
};

#endif
