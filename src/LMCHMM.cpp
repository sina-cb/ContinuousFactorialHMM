#include "LMCHMM.h"
#include <cmath>
#include "Sampler.h"
#include <random>
#include <chrono>
#include <glog/logging.h>
using namespace std;
using namespace google;

LMCHMM::LMCHMM() : LMCHMM(2){
}

LMCHMM::LMCHMM(int layers_count){
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

size_t LMCHMM::learn_hmm_KL(vector<Observation> *observations, double threshold, size_t max_iteration, int N){

    LOG(WARNING) << "HEADER";
    LOG(WARNING) << "Iteration\tKLD_0\tKLD_1\tAVG_KLD";


    if (layers.size() <= 0){
        LOG(FATAL) << "No HMM existing in the layers!!!";
    }

    vector<bool> kl_diverged;
    for (size_t i = 0; i < layers.size(); i++){
        kl_diverged.push_back(false);
    }

    vector< vector<Sample> > test_sample_sets;
    for (size_t i = 0; i < layers.size(); i++){
        vector<Sample> test_samples = ((MCHMM*)layers[i])->get_uniform_samples_from_pi(10000);
        test_sample_sets.push_back(test_samples);
    }

    size_t j = 0;
    for (; j < max_iteration; j++){
        // Get the observations for the first level HMM
        vector<Observation> obs;
        for (size_t i = 0; i < observations->size(); i++){
            obs.push_back((*observations)[i]);
        }

        // Repeat the HMM learning for each level
        vector<double> KLDs;
        for (size_t i = 0; i < layers.size(); i++){
            if (j == 0){
                // Do an EM for the ith level HMM
                ((MCHMM*)layers[i])->learn_hmm(&obs, 1, N);
                ind_initialized[i] = ((MCHMM*)layers[i])->initialized_();
                LOG(INFO) << "EM Finished for HMM in level " << i;
            }else{
                // Get the test_samples from this layer's MCHMM
                DETree * old_gamma = ((MCHMM*)layers[i])->gamma(observations, N).back();

                // Do an EM for the ith level HMM
                ((MCHMM*)layers[i])->learn_hmm(&obs, 1, N);
                ind_initialized[i] = ((MCHMM*)layers[i])->initialized_();
                LOG(INFO) << "EM Finished for HMM in level " << i;

                DETree * new_gamma = ((MCHMM*)layers[i])->gamma(observations, N).back();

                vector<double> estimates_old;
                vector<double> estimates_new;
                double sum_old = 0.0;
                double sum_new = 0.0;
                for (size_t r = 0; r < test_sample_sets[i].size(); r++){
                    estimates_old.push_back(old_gamma->density_value(test_sample_sets[i][r], 0.5));
                    estimates_new.push_back(new_gamma->density_value(test_sample_sets[i][r], 0.5));

                    sum_old += estimates_old.back();
                    sum_new += estimates_new.back();
                }

                // Normalize the density values
                for (size_t r = 0; r < test_sample_sets[i].size(); r++){
                    estimates_old[r] = estimates_old[r] / sum_old;
                    estimates_new[r] = estimates_new[r] / sum_new;
                }

                // Compute the KL divergence factor
                double KLD = ((MCHMM*)layers[i])->KLD_compute(estimates_old, estimates_new);
                LOG(INFO) << "Level: " << i << "\t KLD: " << KLD;
                KLDs.push_back(KLD);

                if (KLD < threshold){
                    kl_diverged[i] = true;
                }

                // Get the most probable sequence of states and use it as the observation for the next state
                if (i < layers.size() - 1){
                    obs = most_probable_seq(observations, i, N);
                    LOG(INFO) << "Got the observation for the next level HMM from HMM in level " << i;
                }
            }
        }

        if (j != 0){
            stringstream ssd;
            ssd << j << "\t";
            double avg_kld = 0.0;
            for (size_t g = 0; g < KLDs.size(); g++){
                ssd << KLDs[g] << "\t";
                avg_kld += KLDs[g];
            }
            avg_kld /= KLDs.size();
            ssd << avg_kld;
            LOG(WARNING) << ssd.str();

            //        bool all_converged = true;
            //        for (size_t i = 0; i < layers.size(); i++){
            //            all_converged = all_converged && kl_diverged[i];
            //        }

            if (avg_kld < threshold){
                LOG(WARNING) << "Iteration: " << j << "\tKL Converged!!!";
                break;
            }else{
                LOG(INFO) << "Iteration: " << j << "\tKL Not Converged. Reseting States!!!";
                for (size_t i = 0; i < layers.size(); i++){
                    kl_diverged[i] = false;
                }
            }
        }
    }

    if (!initialized_()){
        LOG(FATAL) << "Something went wrong in the learning proces!!!";
    }
    return j;
}

size_t LMCHMM::learn_hmm_separately_KL(vector<Observation> *observations, double threshold, size_t max_iteration, int N){
    if (layers.size() <= 0){
        LOG(FATAL) << "No HMM existing in the layers!!!";
    }

    // Get the observations for the first level HMM
    vector<Observation> obs;
    for (size_t i = 0; i < observations->size(); i++){
        obs.push_back((*observations)[i]);
    }

    size_t sum_iterations = 0;
    // Repeat the HMM learning for each level
    for (size_t i = 0; i < layers.size(); i++){
        // Do an EM for the ith level HMM
        ((MCHMM*)layers[i])->learn_hmm_KL(&obs, threshold, max_iteration, N);
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

    return sum_iterations;
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

    LOG(INFO) << "LMCHMM with " << layers.size() << " layers created.";
}

size_t LMCHMM::_layers_count(){
    return layers.size();
}

double LMCHMM::_rho(){
    return rho;
}
