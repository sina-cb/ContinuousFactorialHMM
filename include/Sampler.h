#include <vector>
#include <iostream>
using namespace std;

#ifndef SAMPLER_H
#define SAMPLER_H

class Sampler{

public:

    vector< vector<double> > * likelihood_weighted_sampler(vector< vector<double> > &sample_set);

private:


};

#endif
