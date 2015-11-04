#include "Sampler.h"
#include <tuple>
#include <random>
#include <chrono>
using namespace std;

vector<Sample> * Sampler::likelihood_weighted_sampler(vector<Sample> &sample_set){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(0, sample_set.size());

    vector<Sample> temp = sample_set;

    sample_set.clear();
    for (size_t i = 0; i < temp.size(); i++){
        int index = dist(gen);
        sample_set.push_back(temp[index]);
    }

    return &sample_set;
}

vector<Sample> * Sampler::resample_from(DETree *tree, size_t sample_set_size){
    vector<Sample> * results = new vector<Sample>();
    for (size_t i = 0; i < sample_set_size; i++){
        results->push_back(sample(tree));
    }
    return results;
}

Sample Sampler::sample(DETree *tree){
    Sample *sample = &tree->get_root().samples[0];
    return *sample;
}
