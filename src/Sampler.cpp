#include "Sampler.h"
#include <tuple>
#include <random>
#include <chrono>
using namespace std;

Sample Sampler::sample(DETree *tree){
    Sample sample;
    sample.init_rand(tree->samples_low_limit, tree->samples_high_limit);
    sample.p = 1.0;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    DETreeNode *node = tree->get_root();

    bool cond = !node->leaf_node;
    while(cond){
        int index = node->max_diff_index;
        double max_value = node->max_diff_max_value;
        double min_value = node->max_diff_min_value;

        uniform_real_distribution<double> dist(min_value, max_value);
        double temp = dist(gen);
        sample.values[index] = temp;

        cond = !node->leaf_node;

        if (sample.values[index] < node->cut_value){
            node = node->left_child;
        }else{
            node = node->right_child;
        }
    }

    return sample;
}

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
