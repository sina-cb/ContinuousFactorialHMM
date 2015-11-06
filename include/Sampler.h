#include <vector>
#include <iostream>
#include "Sample.h"
#include "DETree.h"
using namespace std;

#ifndef SAMPLER_H
#define SAMPLER_H

class Sampler{

public:

    Sample likelihood_weighted_sampler(vector<Sample> &sample_set);
    vector<Sample> * likelihood_weighted_resampler(vector<Sample> &sample_set, int size = -1);

    vector<Sample> * resample_from(DETree *tree, size_t sample_set_size);
    Sample sample(DETree* tree);

private:


};

#endif
