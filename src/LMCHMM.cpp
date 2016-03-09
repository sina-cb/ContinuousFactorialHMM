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
    LOG(WARNING) << "LEARN Needs Implementation";
}

vector<DETree> LMCHMM::forward(vector<Observation> *observations, size_t N){
    LOG(WARNING) << "FORWARD Needs Implementation";
    vector<DETree> test;
    return test;
}

vector<Sample> LMCHMM::most_probable_seq(){
    LOG(WARNING) << "MOST PROBABLE Needs Implementation";
    vector<Sample> test;
    return test;
}

void LMCHMM::set_hmm_randomly(int sample_size_pi, int sample_size_m, int sample_size_v, size_t level){
    if (level > layers.size()){
        LOG(FATAL) << "Level is greater than the layers' count!";
        return;
    }

    ((MCHMM*)layers[level])->init_hmm_randomly(sample_size_pi, sample_size_m, sample_size_v);
    ind_initialized[level] = true;
}

void LMCHMM::set_limits(vector<double> *pi_low_limit, vector<double> *pi_high_limit, vector<double> *m_low_limit, vector<double> *m_high_limit,
                        vector<double> *v_low_limit, vector<double> *v_high_limit, size_t level){
    if (level > layers.size()){
        LOG(FATAL) << "Level is greater than the layers' count!";
        return;
    }

    ((MCHMM*)layers[level])->set_limits(pi_low_limit, pi_high_limit, m_low_limit, m_high_limit, v_low_limit, v_high_limit);
}

void LMCHMM::set_distributions(vector<Sample> *pi, vector<Sample> *m, vector<Sample> *v, double rho, size_t level){
    if (level > layers.size()){
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
