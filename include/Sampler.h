#include <vector>
#include <iostream>
#include "Sample.h"
#include "DETree.h"
using namespace std;

#ifndef SAMPLER_H
#define SAMPLER_H

class Sampler{

public:

    vector<Sample> * likelihood_weighted_sampler(vector<Sample> &sample_set);

    vector<Sample> * resample_from(DETree *tree, size_t sample_set_size);
    Sample sample(DETree* tree);

private:


};

#endif
