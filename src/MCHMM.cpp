#include "MCHMM.h"
#include <cmath>
#include "Sampler.h"
#include <random>
#include <chrono>
#include <glog/logging.h>
#include <cassert>
#include <boost/math/distributions/students_t.hpp>
using namespace std;
using namespace google;

MCHMM::MCHMM(){
    pi = new vector<Sample>();
    m = new vector<Sample>();
    v = new vector<Sample>();
}

/**
 * @brief MCHMM::forward Goes one step forward in time according to the HMM distributions
 * @param observations Observations needed to do the reasoning based on the HMM
 * @param N Number of samples used in the resampling step of any sampling inside this method
 * @return Alpha DETree for the next time step distribution
 */
DETree * MCHMM::forward(vector<Observation> *observations, size_t N){
    vector<Sample> alpha_samples[2];
    Sampler sampler;
    size_t T = observations->size();

    alpha_samples[0] = sampler.resample_from(pi_tree, N);

    // STEP 2
    size_t t = 0;
    for (t = 1; t < T; t++){
        // STEP 2(a)
        vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[(t - 1) % 2], N);
        double sum_densities = 0.0;

        for (size_t i = 0; i < temp.size(); i++){
            // STEP 2(b)
            Sample x = sampler.sample_given(m_tree, temp[i]);

            for (size_t i = 0; i < temp[i].size(); i++){
                x.values.pop_back();
            }

            // STEP 2(c)
            Sample v_temp = (*observations)[t].combine(x);
            double density = v_tree->density_value(v_temp, rho);
            x.p = density;

            sum_densities += density;
            temp[i] = x;
        }

        // Normalizing the probabilities
        for (size_t i = 0; i < temp.size(); i++){
            temp[i].p = temp[i].p / sum_densities;
        }

        // STEP 2(d)
        alpha_samples[t % 2] = temp;
    }

    vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[(t - 1) % 2], N);
    for (size_t i = 0; i < temp.size(); i++){
        // STEP 2(b)
        Sample x = sampler.sample_given(m_tree, temp[i]);
        x.p = 1.0 / temp.size();
        temp[i] = x;
    }

    return new DETree(temp, pi_low_limit, pi_high_limit);
}

/**
 * @brief MCHMM::gamma It perform the forward-backward algorithms and combine the results to get the
 * gamma distribution, it is useful mostly for getting the most probable states
 * @param observations Observations needed to do the reasoning based on the HMM
 * @param N Number of samples used in the resampling step of any sampling inside this method
 * @return Gamma DETrees for each observation
 */
vector<DETree *> MCHMM::gamma(vector<Observation> *observations, size_t N){
    Sampler sampler;
    size_t T = observations->size();

    vector<Sample> alpha_samples[2];
    vector<Sample> beta_samples[2];

    vector<DETree*> alpha_trees;
    vector<DETree*> beta_trees;
    vector<DETree*> gamma_trees;

    // STEP 1
    alpha_samples[0] = sampler.resample_from(pi_tree, N);
    alpha_trees.push_back(new DETree(alpha_samples[0], pi_low_limit, pi_high_limit));

    // STEP 2
    for (size_t t = 1; t < T; t++){
        // STEP 2(a)
        vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[(t - 1) % 2], N);
        double sum_densities = 0.0;

        for (size_t i = 0; i < temp.size(); i++){
            // STEP 2(b)
            Sample x = sampler.sample_given(m_tree, temp[i]);

            for (size_t i = 0; i < temp[i].size(); i++){
                x.values.pop_back();
            }

            // STEP 2(c)
            Sample v_temp = (*observations)[t].combine(x);
            double density = v_tree->density_value(v_temp, rho);
            x.p = density;

            sum_densities += density;
            temp[i] = x;
        }

        // Normalizing the probabilities
        for (size_t i = 0; i < temp.size(); i++){
            temp[i].p = temp[i].p / sum_densities;
        }

        // STEP 2(d)
        alpha_samples[t % 2] = temp;
        alpha_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
    }

    // STEP 3
    beta_samples[0] = sampler.uniform_sampling(pi_low_limit, pi_high_limit, N);
    beta_trees.push_back(new DETree(beta_samples[0], pi_low_limit, pi_high_limit));

    // STEP 4
    for (size_t t = T - 1; t >= 1; t--){
        // STEP 4(a)
        int index_t = ((T) - (t + 1)) % 2;
        vector<Sample> temp = sampler.likelihood_weighted_resampler(beta_samples[index_t], N);
        double sum_densities = 0.0;

        for (size_t i = 0; i < temp.size(); i++){
            // STEP 4(b)
            Sample x = sampler.sample_given(m_tree, temp[i]);

            for (size_t i = 0; i < temp[i].size(); i++){
                x.values.pop_back();
            }

            // STEP 4(c)
            Sample v_temp = (*observations)[t].combine(x);
            double density = v_tree->density_value(v_temp, rho);
            x.p = density;

            sum_densities += density;
            temp[i] = x;
        }

        // Normalizing the probabilities
        for (size_t i = 0; i < temp.size(); i++){
            temp[i].p = temp[i].p / sum_densities;
        }

        // STEP 4(d)
        beta_samples[(index_t + 1) % 2] = temp;
        beta_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
    }

    // STEP 5
    for (size_t t = 0; t < T; t++){
        vector<Sample> temp;
        double sum_density = 0.0;

        int index_t = (T) - (t + 1);

        // STEP 5(a)
        for (size_t j = 0; j < N / 2; j++){
            Sample sample = sampler.sample(alpha_trees[t]);
            sample.p = (*beta_trees[index_t]).density_value(sample, rho);
            sum_density += sample.p;
            temp.push_back(sample);
        }

        // STEP 5(b)
        for (size_t j = 0; j < N - (N / 2); j++){
            Sample sample = sampler.sample(beta_trees[index_t]);
            sample.p = (*alpha_trees[t]).density_value(sample, rho);
            sum_density += sample.p;
            temp.push_back(sample);
        }

        // Normalizing the probabilities
        for (size_t i = 0; i < temp.size(); i++){
            temp[i].p = temp[i].p / sum_density;
        }

        gamma_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
    }

    alpha_samples[0].clear();
    alpha_samples[1].clear();
    beta_samples[0].clear();
    beta_samples[1].clear();

    for (size_t i = 0; i < alpha_trees.size(); i++){
        delete alpha_trees[i];
    }

    for (size_t i = 0; i < beta_trees.size(); i++){
        delete beta_trees[i];
    }

    assert(gamma_trees.size() == T);

    return gamma_trees;
}

/**
 * @brief MCHMM::learn_hmm it takes in some observations and perform the EM as many iterations as the max_iteration arg
 * @param observations Observations needed for learning the HMM distributions
 * @param max_iteration Maximum number of iterations performed for the EM
 * @param N Number of samples used in the resampling step of any sampling inside this method
 */
void MCHMM::learn_hmm_KL(vector<Observation> *observations, double threshold, size_t max_iteration, int N){
    if (observations->size() < 2){
        LOG(ERROR) << "Not enough observation data!";
        return;
    }

    Sampler sampler;
    size_t T = observations->size();

    if (pi->size() < 1 || v->size() < 1 || m->size() < 1){
        LOG(INFO) << "Init HMM Randomly!";
        init_hmm_randomly(N, N, N);
    }

    bool cond = true;
    size_t iteration = 0;
    DETree* old_gamma_tree = NULL;

    vector<Sample> test_samples = sampler.uniform_sampling(pi_low_limit, pi_high_limit, 1000);

    while (cond){
        vector<Sample> alpha_samples[2];
        vector<Sample> beta_samples[2];

        vector<DETree*> alpha_trees;
        vector<DETree*> beta_trees;
        vector<DETree*> gamma_trees;

        /////////////////E STEP/////////////////
        {
            // STEP 1
            alpha_samples[0] = sampler.resample_from(pi_tree, N);
            alpha_trees.push_back(new DETree(alpha_samples[0], pi_low_limit, pi_high_limit));

            // STEP 2
            for (size_t t = 1; t < T; t++){
                // STEP 2(a)
                vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[(t - 1) % 2], N);
                double sum_densities = 0.0;

                for (size_t i = 0; i < temp.size(); i++){
                    // STEP 2(b)
                    Sample x = sampler.sample_given(m_tree, temp[i]);

                    for (size_t i = 0; i < temp[i].size(); i++){
                        x.values.pop_back();
                    }

                    // STEP 2(c)
                    Sample v_temp = (*observations)[t].combine(x);
                    double density = v_tree->density_value(v_temp, rho);
                    x.p = density;

                    sum_densities += density;
                    temp[i] = x;
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_densities;
                }

                // STEP 2(d)
                alpha_samples[t % 2] = temp;
                alpha_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
            }

            // STEP 3
            beta_samples[0] = sampler.uniform_sampling(pi_low_limit, pi_high_limit, N);
            beta_trees.push_back(new DETree(beta_samples[0], pi_low_limit, pi_high_limit));

            // STEP 4
            for (size_t t = T - 1; t >= 1; t--){
                // STEP 4(a)
                int index_t = ((T) - (t + 1)) % 2;
                vector<Sample> temp = sampler.likelihood_weighted_resampler(beta_samples[index_t], N);
                double sum_densities = 0.0;

                for (size_t i = 0; i < temp.size(); i++){
                    // STEP 4(b)
                    Sample x = sampler.sample_given(m_tree, temp[i]);

                    for (size_t i = 0; i < temp[i].size(); i++){
                        x.values.pop_back();
                    }

                    // STEP 4(c)
                    Sample v_temp = (*observations)[t].combine(x);
                    double density = v_tree->density_value(v_temp, rho);
                    x.p = density;

                    sum_densities += density;
                    temp[i] = x;
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_densities;
                }

                // STEP 4(d)
                beta_samples[(index_t + 1) % 2] = temp;
                beta_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
            }

            // STEP 5
            for (size_t t = 1; t < T; t++){
                vector<Sample> temp;
                double sum_density = 0.0;

                int index_t = (T) - (t);

                // STEP 5(a)
                for (int j = 0; j < N / 2; j++){
                    Sample sample = sampler.sample(alpha_trees[t]);
                    sample.p = (*beta_trees[index_t]).density_value(sample, rho);
                    sum_density += sample.p;
                    temp.push_back(sample);
                }

                // STEP 5(b)
                for (int j = 0; j < N - (N / 2); j++){
                    Sample sample = sampler.sample(beta_trees[index_t]);
                    sample.p = (*alpha_trees[t]).density_value(sample, rho);
                    sum_density += sample.p;
                    temp.push_back(sample);
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_density;
                }

                gamma_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
            }

            alpha_samples[0].clear();
            alpha_samples[1].clear();
            beta_samples[0].clear();
            beta_samples[1].clear();

            LOG(INFO) << "End of E Step at iteration: " << iteration;
        }

        /////////////////M STEP/////////////////
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            vector<Sample> * temp_m = new vector<Sample>();
            vector<Sample> * temp_v = new vector<Sample>();
            vector<Sample> * temp_pi = new vector<Sample>();

            // STEP 1
            for (int i = 0; i < N; i++){
                uniform_real_distribution<double> dist(1, T - 2);
                int t = dist(gen);

                Sample x = sampler.sample(gamma_trees[t]);
                Sample x_prime = sampler.sample(gamma_trees[t + 1]);

                Sample temp = x.combine(x_prime.values);
                temp.p = 1.0 / N;
                temp_m->push_back(temp);
            }

            // STEP 2
            for (int i = 0; i < N; i++){
                uniform_real_distribution<double> dist(1, T - 1);
                int t = dist(gen);

                Sample x = sampler.sample(gamma_trees[t]);

                Sample temp = (*observations)[t].combine(x);
                temp.p = 1.0 / N;
                temp_v->push_back(temp);
            }

            // STEP 3
            int pi_size = pi->size();
            for (int i = 0; i < pi_size; i++){
                temp_pi->push_back(sampler.sample(gamma_trees[0]));
            }

            delete pi;
            pi = temp_pi;
            pi_tree->create_tree(*pi, pi_low_limit, pi_high_limit);

            delete m;
            m =  temp_m;
            m_tree->create_tree(*m, m_low_limit, m_high_limit);

            delete v;
            v = temp_v;
            v_tree->create_tree(*v, v_low_limit, v_high_limit);

            LOG(INFO) << "End of M Step at iteration: " << iteration;

        }

        /////////////////ANNEALING/////////////////
        if (rho > 0.01)
            rho = rho * rho_bar;

        /////////////////SAMPLE SET SIZE/////////////////
        if (N < (int)max_sample_size)
            N = N; // * eta;

        /////////////////STOP CONDITION/////////////////
        {
            LOG(INFO) << "Iteration " << iteration + 1 << " Finished!" << "\n";
            iteration++;
            if (iteration >= max_iteration){
                cond = false;
            }

            if (iteration > 1){ // Do the KL if we have at least on previous HMM parameters set!!!
                // Generate a lot of samples uniformly

                // Find the density estimations for each generated sample
                vector<double> estimates_old;
                vector<double> estimates_new;
                double sum_old = 0.0;
                double sum_new = 0.0;
                for (size_t r = 0; r < test_samples.size(); r++){
                    estimates_old.push_back(old_gamma_tree->density_value(test_samples[r], 0.5));
                    estimates_new.push_back(gamma_trees.back()->density_value(test_samples[r], 0.5));

                    sum_old += estimates_old.back();
                    sum_new += estimates_new.back();
                }

                // Normalize the density values
                for (size_t r = 0; r < test_samples.size(); r++){
                    estimates_old[r] = estimates_old[r] / sum_old;
                    estimates_new[r] = estimates_new[r] / sum_new;
                }

                // Compute the KL divergence factor
                double KLD = KLD_compute(estimates_old, estimates_new);

                LOG(ERROR) << "KLD: " << KLD;

                // If KLD < threshold --> STOP
                if (KLD < threshold){
                    cond = false;
                }
            }
        }

        for (size_t i = 0; i < alpha_trees.size(); i++){
            delete alpha_trees[i];
        }

        for (size_t i = 0; i < beta_trees.size(); i++){
            delete beta_trees[i];
        }

        for (size_t i = 0; i < gamma_trees.size() - 1; i++){
            delete gamma_trees[i];
        }

        if (old_gamma_tree){
            delete old_gamma_tree;
        }

        old_gamma_tree = gamma_trees.back();
    }

    initialized = true;
}

/**
 * @brief MCHMM::KLD_compute computes the Kullback-Leibler divergence of two distributions
 * @param P The true distribution (in this application, the true is our old distribution)
 * @param Q The estimated distribution (in this application, the estimated is our new distribution)
 * @return KLD value
 */
double MCHMM::  KLD_compute(vector<double> P, vector<double> Q){
    double KLD = 0.0;
    for (size_t i = 0; i < P.size(); i++){
        KLD += P[i] * std::log(P[i] / Q[i]);
    }
    return KLD;
}

vector<Sample> MCHMM::get_uniform_samples_from_pi(size_t N){
    Sampler sampler;
    return sampler.uniform_sampling(pi_low_limit, pi_high_limit, N);
}

/**
 * @brief MCHMM::learn_hmm it takes in some observations and perform the EM as many iterations as the max_iteration arg
 * @param observations Observations needed for learning the HMM distributions
 * @param max_iteration Maximum number of iterations performed for the EM
 * @param N Number of samples used in the resampling step of any sampling inside this method
 */
void MCHMM::learn_hmm(vector<Observation> *observations, size_t max_iteration, int N){

    if (observations->size() < 2){
        LOG(ERROR) << "Not enough observation data!";
        return;
    }

    Sampler sampler;
    size_t T = observations->size();

    if (pi->size() < 1 || v->size() < 1 || m->size() < 1){
        LOG(INFO) << "Init HMM Randomly!";
        init_hmm_randomly(N, N, N);
    }

    bool cond = true;
    size_t iteration = 0;

    while (cond){
        vector<Sample> alpha_samples[2];
        vector<Sample> beta_samples[2];

        vector<DETree*> alpha_trees;
        vector<DETree*> beta_trees;
        vector<DETree*> gamma_trees;

        /////////////////E STEP/////////////////
        {
            // STEP 1
            alpha_samples[0] = sampler.resample_from(pi_tree, N);
            alpha_trees.push_back(new DETree(alpha_samples[0], pi_low_limit, pi_high_limit));

            // STEP 2
            for (size_t t = 1; t < T; t++){
                // STEP 2(a)
                vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[(t - 1) % 2], N);
                double sum_densities = 0.0;

                for (size_t i = 0; i < temp.size(); i++){
                    // STEP 2(b)
                    Sample x = sampler.sample_given(m_tree, temp[i]);

                    for (size_t i = 0; i < temp[i].size(); i++){
                        x.values.pop_back();
                    }

                    // STEP 2(c)
                    Sample v_temp = (*observations)[t].combine(x);
                    double density = v_tree->density_value(v_temp, rho);
                    x.p = density;

                    sum_densities += density;
                    temp[i] = x;
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_densities;
                }

                // STEP 2(d)
                alpha_samples[t % 2] = temp;
                alpha_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
            }

            // STEP 3
            beta_samples[0] = sampler.uniform_sampling(pi_low_limit, pi_high_limit, N);
            beta_trees.push_back(new DETree(beta_samples[0], pi_low_limit, pi_high_limit));

            // STEP 4
            for (size_t t = T - 1; t >= 1; t--){
                // STEP 4(a)
                int index_t = ((T) - (t + 1)) % 2;
                vector<Sample> temp = sampler.likelihood_weighted_resampler(beta_samples[index_t], N);
                double sum_densities = 0.0;

                for (size_t i = 0; i < temp.size(); i++){
                    // STEP 4(b)
                    Sample x = sampler.sample_given(m_tree, temp[i]);

                    for (size_t i = 0; i < temp[i].size(); i++){
                        x.values.pop_back();
                    }

                    // STEP 4(c)
                    Sample v_temp = (*observations)[t].combine(x);
                    double density = v_tree->density_value(v_temp, rho);
                    x.p = density;

                    sum_densities += density;
                    temp[i] = x;
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_densities;
                }

                // STEP 4(d)
                beta_samples[(index_t + 1) % 2] = temp;
                beta_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
            }

            // STEP 5
            for (size_t t = 1; t < T; t++){
                vector<Sample> temp;
                double sum_density = 0.0;

                int index_t = (T) - (t);

                // STEP 5(a)
                for (int j = 0; j < N / 2; j++){
                    Sample sample = sampler.sample(alpha_trees[t]);
                    sample.p = (*beta_trees[index_t]).density_value(sample, rho);
                    sum_density += sample.p;
                    temp.push_back(sample);
                }

                // STEP 5(b)
                for (int j = 0; j < N - (N / 2); j++){
                    Sample sample = sampler.sample(beta_trees[index_t]);
                    sample.p = (*alpha_trees[t]).density_value(sample, rho);
                    sum_density += sample.p;
                    temp.push_back(sample);
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_density;
                }

                gamma_trees.push_back(new DETree(temp, pi_low_limit, pi_high_limit));
            }

            alpha_samples[0].clear();
            alpha_samples[1].clear();
            beta_samples[0].clear();
            beta_samples[1].clear();

            LOG(INFO) << "End of E Step at iteration: " << iteration;
        }

        /////////////////M STEP/////////////////
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            vector<Sample> * temp_m = new vector<Sample>();
            vector<Sample> * temp_v = new vector<Sample>();
            vector<Sample> * temp_pi = new vector<Sample>();

            // STEP 1
            for (int i = 0; i < N; i++){
                uniform_real_distribution<double> dist(1, T - 2);
                int t = dist(gen);

                Sample x = sampler.sample(gamma_trees[t]);
                Sample x_prime = sampler.sample(gamma_trees[t + 1]);

                Sample temp = x.combine(x_prime.values);
                temp.p = 1.0 / N;
                temp_m->push_back(temp);
            }

            // STEP 2
            for (int i = 0; i < N; i++){
                uniform_real_distribution<double> dist(1, T - 1);
                int t = dist(gen);

                Sample x = sampler.sample(gamma_trees[t]);

                Sample temp = (*observations)[t].combine(x);
                temp.p = 1.0 / N;
                temp_v->push_back(temp);
            }

            // STEP 3
            int pi_size = pi->size();
            for (int i = 0; i < pi_size; i++){
                temp_pi->push_back(sampler.sample(gamma_trees[0]));
            }

            delete pi;
            pi = temp_pi;
            pi_tree->create_tree(*pi, pi_low_limit, pi_high_limit);

            delete m;
            m =  temp_m;
            m_tree->create_tree(*m, m_low_limit, m_high_limit);

            delete v;
            v = temp_v;
            v_tree->create_tree(*v, v_low_limit, v_high_limit);

            LOG(INFO) << "End of M Step at iteration: " << iteration;

        }

        /////////////////ANNEALING/////////////////
        if (rho > 0.01)
            rho = rho * rho_bar;

        /////////////////SAMPLE SET SIZE/////////////////
        if (N < (int)max_sample_size)
            N = N; // * eta;

        /////////////////STOP CONDITION/////////////////
        LOG(INFO) << "Iteration " << iteration + 1 << " Finished!" << "\n";
        iteration++;
        if (iteration >= max_iteration){
            cond = false;
        }

        for (size_t i = 0; i < alpha_trees.size(); i++){
            delete alpha_trees[i];
        }

        for (size_t i = 0; i < beta_trees.size(); i++){
            delete beta_trees[i];
        }

        for (size_t i = 0; i < gamma_trees.size(); i++){
            delete gamma_trees[i];
        }
    }

    initialized = true;
}

/**
 * @brief MCHMM::set_distributions Instead of learning the distributions one can use this to prime the HMM with pre-collected samples
 * @param pi Samples collected for the initial state distribution PI
 * @param m Samples collected for the transition distribution M
 * @param v Samples collected for the observation distribution NU
 * @param rho Amount of effect that different levels of the DETree have on the computed density value (default: 0.5)
 */
void MCHMM::set_distributions(vector<Sample> *pi, vector<Sample> *m, vector<Sample> *v, double rho){
    this->pi = new vector<Sample>();
    this->m = new vector<Sample>();
    this->v = new vector<Sample>();

    for (size_t i = 0; i < pi->size(); i++){
        this->pi->push_back((*pi)[i]);
    }

    for (size_t i = 0; i < m->size(); i++){
        this->m->push_back((*m)[i]);
    }

    for (size_t i = 0; i < v->size(); i++){
        this->v->push_back((*v)[i]);
    }

    this->rho = rho;

    pi_tree = new DETree(*pi, pi_low_limit, pi_high_limit);
    m_tree = new DETree(*m, m_low_limit, m_high_limit);
    v_tree = new DETree(*v, v_low_limit, v_high_limit);

    initialized = true;
}

void MCHMM::set_limits(vector<double> *pi_low_limit, vector<double> *pi_high_limit,
                       vector<double> *m_low_limit, vector<double> *m_high_limit,
                       vector<double> *v_low_limit, vector<double> *v_high_limit
                       )
{
    this->pi_low_limit = pi_low_limit;
    this->pi_high_limit = pi_high_limit;

    this->m_low_limit = m_low_limit;
    this->m_high_limit = m_high_limit;

    this->v_low_limit = v_low_limit;
    this->v_high_limit = v_high_limit;
}

void MCHMM::init_hmm_randomly(int sample_size_pi, int sample_size_m, int sample_size_v){

    if (pi_low_limit == NULL){
        LOG(FATAL) << "Please set the limits first and then run this method!!!";
    }

    for (int i = 0; i < sample_size_pi; i++){
        Sample sample;
        sample.init_rand(pi_low_limit, pi_high_limit);
        sample.p = 1.0 / sample_size_pi;
        pi->push_back(sample);
    }

    for (int i = 0; i < sample_size_m; i++){
        Sample sample;
        sample.init_rand(m_low_limit, m_high_limit);
        sample.p = 1.0 / sample_size_m;
        m->push_back(sample);
    }

    for (int i = 0; i < sample_size_v; i++){
        Sample sample;
        sample.init_rand(v_low_limit, v_high_limit);
        sample.p = 1.0 / sample_size_v;
        v->push_back(sample);
    }

    pi_tree = new DETree(*pi, pi_low_limit, pi_high_limit);
    m_tree = new DETree(*m, m_low_limit, m_high_limit);
    v_tree = new DETree(*v, v_low_limit, v_high_limit);

    initialized = true;
}

double MCHMM::_rho(){
    return this->rho;
}

bool MCHMM::initialized_(){
    return initialized;
}

DETree* MCHMM::pi_tree_(){
    return pi_tree;
}
