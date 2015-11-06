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

vector<Sample> * Sampler::likelihood_weighted_resampler(vector<Sample> &sample_set, int size){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(0, 1.0);

    if (size == -1){
        size = sample_set.size();
    }

    vector<Sample> temp = sample_set;

    vector<double> low_p;
    vector<double> high_p;

    low_p.push_back(0.0);
    high_p.push_back(temp[0].p);
    for (size_t i = 1; i < temp.size(); i++){
        low_p.push_back(high_p[i - 1]);
        high_p.push_back(high_p[i - 1] + temp[i].p);
    }

    sample_set.clear();
    for (size_t i = 0; i < size; i++){
        double temp_p = dist(gen);
        int index = 0;
        for (size_t j = 0; j < low_p.size(); j++){
            if (temp_p >= low_p[j] && temp_p <= high_p[j]){
                index = j;
                break;
            }
        }

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
