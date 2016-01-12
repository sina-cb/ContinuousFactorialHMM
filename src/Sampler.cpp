#include "Sampler.h"
#include <tuple>
#include <random>
#include <chrono>
using namespace std;

Sample Sampler::sample_given(DETree *tree, Sample &given){
    Sample sample;
    sample.init_rand(tree->samples_low_limit, tree->samples_high_limit);
    sample.p = 1.0;

    for (size_t i = given.size(); i < sample.size(); i++){
        sample.values[i] = given.values[i - given.size()];
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    DETreeNode *node = tree->get_root();

    bool cond = !node->leaf_node;
    while(cond){
        int index = node->max_diff_index;

        if (index < (int)given.size()){
            double max_value = node->max_diff_max_value;
            double min_value = node->max_diff_min_value;

            uniform_real_distribution<double> dist(min_value, max_value);
            double temp = dist(gen);
            sample.values[index] = temp;
        }

        cond = !node->leaf_node;
        if (sample.values[index] < node->cut_value){
            node = node->left_child;
        }else{
            node = node->right_child;
        }
    }

    Sample result;
    for (int i = 0; i < (int)given.size(); i++){
        result.values.push_back(sample.values[i]);
    }

    return result;
}

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

Sample Sampler::likelihood_weighted_sampler(vector<Sample> &sample_set){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(0, 1.0);

    vector<double> low_p;
    vector<double> high_p;

    low_p.push_back(0.0);
    high_p.push_back(sample_set[0].p);
    for (size_t i = 1; i < sample_set.size(); i++){
        low_p.push_back(high_p[i - 1]);
        high_p.push_back(high_p[i - 1] + sample_set[i].p);
    }

    double temp_p = dist(gen);
    int index = 0;
    for (size_t j = 0; j < low_p.size(); j++){
        if (temp_p >= low_p[j] && temp_p <= high_p[j]){
            index = j;
            break;
        }
    }

    return sample_set[index];
}

vector<Sample> Sampler::uniform_sampling(vector<double> *sample_low, vector<double> *sample_high, size_t N){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);

    vector<Sample> results;

    for (size_t i = 0; i < N; i++){
        Sample sample;
        for (size_t j = 0; j < sample_low->size(); j++){
            uniform_real_distribution<double> dist((*sample_low)[j], (*sample_high)[j]);
            sample.values.push_back(dist(gen));
        }
        results.push_back(sample);
    }

    return results;
}

vector<Sample> Sampler::likelihood_weighted_resampler(vector<Sample> &sample_set, int size){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(0, 1.0);

    if (size == -1){
        size = sample_set.size();
    }

    vector<double> low_p;
    vector<double> high_p;

    low_p.push_back(0.0);
    high_p.push_back(sample_set[0].p);
    for (size_t i = 1; i < sample_set.size(); i++){
        low_p.push_back(high_p[i - 1]);
        high_p.push_back(high_p[i - 1] + sample_set[i].p);
    }

    vector<Sample> temp;
    for (int i = 0; i < size; i++){
        double temp_p = dist(gen);
        int index = 0;
        for (size_t j = 0; j < low_p.size(); j++){
            if (temp_p >= low_p[j] && temp_p <= high_p[j]){
                index = j;
                break;
            }
        }

        if (index >= (int)sample_set.size()) index = sample_set.size() - 1;
        temp.push_back(sample_set[index]);
    }

    return temp;
}

vector<Sample> Sampler::resample_from(DETree *tree, size_t sample_set_size){
    vector<Sample> results;
    for (size_t i = 0; i < sample_set_size; i++){
        Sample smp = sample(tree);
        smp.p = 1.0 / sample_set_size;
        results.push_back(smp);
    }
    return results;
}
