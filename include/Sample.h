#include <vector>
#include <iostream>
#include <string>
#include <sstream>
using namespace std;

#ifndef SAMPLE_H
#define SAMPLE_H

class Sample{

public:
    vector<double> values;
    double p;

    void init_rand(vector<double> *low_limit, vector<double> *high_limit);
    Sample combine(vector<double> second);
    size_t size();
    string str();
};

#endif
