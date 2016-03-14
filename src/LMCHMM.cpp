#include "LMCHMM.h"
#include <cmath>
#include "Sampler.h"
#include <random>
#include <chrono>
#include <glog/logging.h>
using namespace std;
using namespace google;

LMCHMM::LMCHMM(){

}

LMCHMM::LMCHMM(int layers_count) : LMCHMM(){
    set_layers(layers_count);
}

void LMCHMM::learn_hmm(vector<Observation> *observations, size_t max_iteration, int N){

    if (layers.size() <= 0){
        LOG(FATAL) << "No HMM existing in the layers!!!";
    }

    for (size_t j = 0; j < max_iteration; j++){
        // Get the observations for the first level HMM
        vector<Observation> obs;
        for (size_t i = 0; i < observations->size(); i++){
            obs.push_back((*observations)[i]);
        }

        // Repeat the HMM learning for each level
        for (size_t i = 0; i < layers.size(); i++){
            // Do an EM for the ith level HMM
            ((MCHMM*)layers[i])->learn_hmm(&obs, 1, N);
            ind_initialized[i] = ((MCHMM*)layers[i])->initialized_();
            LOG(INFO) << "EM Finished for HMM in level " << i;

            // Get the most probable sequence of states and use it as the observation for the next state
            if (i < layers.size() - 1){
                obs = most_probable_seq(observations, i, N);
                LOG(INFO) << "Got the observation for the next level HMM from HMM in level " << i;
            }
        }
    }

    if (!initialized_()){
        LOG(FATAL) << "Something went wrong in the learning proces!!!";
    }
}

void LMCHMM::learn_hmm_separately(vector<Observation> *observations, size_t max_iteration, int N){

    if (layers.size() <= 0){
        LOG(FATAL) << "No HMM existing in the layers!!!";
    }


    // Get the observations for the first level HMM
    vector<Observation> obs;
    for (size_t i = 0; i < observations->size(); i++){
        obs.push_back((*observations)[i]);
    }

    // Repeat the HMM learning for each level
    for (size_t i = 0; i < layers.size(); i++){
        // Do an EM for the ith level HMM
        ((MCHMM*)layers[i])->learn_hmm(&obs, max_iteration, N);
        ind_initialized[i] = ((MCHMM*)layers[i])->initialized_();
        LOG(INFO) << "EM Finished for HMM in level " << i;

        // Get the most probable sequence of states and use it as the observation for the next state
        if (i < layers.size() - 1){
            obs = most_probable_seq(observations, i, N);
            LOG(INFO) << "Got the observation for the next level HMM from HMM in level " << i;
        }
    }


    if (!initialized_()){
        LOG(FATAL) << "Something went wrong in the learning proces!!!";
    }
}

vector<DETree *> LMCHMM::forward(vector<Observation> *observations, size_t N){
    if (!initialized_()){
        LOG(FATAL) << "Not all HMMs in layers are initialized!";
    }
    vector<DETree *> results;

    MCHMM * temp_hmm = (MCHMM*)layers[0];
    DETree * alpha_tree = temp_hmm->forward(observations, N);
    results.push_back(alpha_tree);

    vector<Observation> obs;
    for (size_t i = 0; i < observations->size(); i++){
        obs.push_back((*observations)[i]);
    }

    for (size_t i = 1; i < layers.size(); i++){
        obs = most_probable_seq(&obs, i, N);
        temp_hmm = (MCHMM*)layers[i];
        alpha_tree = temp_hmm->forward(&obs, N);
        results.push_back(alpha_tree);
    }

    return results;
}

vector<Observation> LMCHMM::most_probable_seq(vector<Observation> * observations, size_t level, int N){
    if (level >= layers.size()){
        LOG(FATAL) << "Level is greater than the layers' count!";
    }

    vector<DETree*> trees;
    Sampler sampler;

    MCHMM * temp_hmm = (MCHMM*)layers[level];
    trees = temp_hmm->gamma(observations, N);

    vector<Observation> observations_temp;
    for (size_t i = 0; i < trees.size(); i++){
        Sample sample = sampler.sample(trees[i]);
        Observation state;
        state = state.convert(sample);
        observations_temp.push_back(state);
    }

    for (size_t i = 0; i < trees.size(); i++){
        delete trees[i];
    }

    return observations_temp;
}

void LMCHMM::set_hmm_randomly(int sample_size_pi, int sample_size_m, int sample_size_v, size_t level){
    if (level >= layers.size()){
        LOG(FATAL) << "Level is greater than the layers' count!";
        return;
    }

    ((MCHMM*)layers[level])->init_hmm_randomly(sample_size_pi, sample_size_m, sample_size_v);
    ind_initialized[level] = true;
}

void LMCHMM::set_limits(vector<double> *pi_low_limit, vector<double> *pi_high_limit, vector<double> *m_low_limit, vector<double> *m_high_limit,
                        vector<double> *v_low_limit, vector<double> *v_high_limit, size_t level){
    if (level >= layers.size()){
        LOG(FATAL) << "Level is greater than the layers' count!";
        return;
    }

    ((MCHMM*)layers[level])->set_limits(pi_low_limit, pi_high_limit, m_low_limit, m_high_limit, v_low_limit, v_high_limit);
}

void LMCHMM::set_distributions(vector<Sample> *pi, vector<Sample> *m, vector<Sample> *v, double rho, size_t level){
    if (level >= layers.size()){
        LOG(FATAL) << "Level is greater than the layers' count!";
        return;
    }

    ((MCHMM*)layers[level])->set_distributions(pi, m, v, rho);
    ind_initialized[level] = true;
}

bool LMCHMM::initialized_(){
    if (ind_initialized.size() <= 0){
        return false;
    }
    for (size_t i = 0; i < ind_initialized.size(); i++){
        if (ind_initialized[i] == false){
            return false;
        }
    }
    return true;
}

void LMCHMM::set_layers(size_t layers_count){
    for (size_t i = 0; i < layers_count; i++){
        layers.push_back(new MCHMM());
        ind_initialized.push_back(false);
    }
}

size_t LMCHMM::_layers_count(){
    return layers.size();
}

double LMCHMM::_rho(){
    return rho;
}
