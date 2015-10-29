#include "Sampler.h"
#include <tuple>
#include <random>
#include <chrono>
using namespace std;

vector< vector<double> > * Sampler::likelihood_weighted_sampler(vector<vector<double> > &sample_set){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(0, sample_set.size());

    vector< vector<double> > temp = sample_set;

    sample_set.clear();
    for (int i = 0; i < temp.size(); i++){
        int index = dist(gen);
        sample_set.push_back(temp[index]);
    }

    return &sample_set;
}
