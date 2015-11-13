#include "MCFHMM.h"
#include <cmath>
#include "Sampler.h"
#include <random>
#include <chrono>
#include <glog/logging.h>
using namespace std;

MCFHMM::MCFHMM(){
    pi = new vector<Sample>();
    m = new vector<Sample>();
    v = new vector<Sample>();
}

vector<vector<Sample> > MCFHMM::forward(vector<Observation> *observations, size_t N){
    vector<vector<Sample> > alpha_samples;
    Sampler sampler;
    size_t T = observations->size();

    alpha_samples.push_back(sampler.resample_from(pi_tree, N));

    // STEP 2
    for (size_t t = 1; t <= T; t++){
        // STEP 2(a)
        vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[t - 1], N);
        double sum_densities = 0.0;

        for (size_t i = 0; i < temp.size(); i++){
            // STEP 2(b)
            Sample x = sampler.sample_given(m_tree, temp[i]);

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
        alpha_samples.push_back(temp);
    }

    return alpha_samples;
}

void MCFHMM::learn_hmm(vector<Observation> *observations, size_t max_iteration, int N){
    Sampler sampler;
    size_t T = observations->size();

    init_hmm(N, N, N);

    bool cond = true;
    size_t iteration = 0;

    while (cond){
        vector<vector<Sample> > alpha_samples;
        vector<vector<Sample> > beta_samples;
        vector<vector<Sample> > gamma_samples;

        vector<DETree> alpha_trees;
        vector<DETree> beta_trees;
        vector<DETree> gamma_trees;

        /////////////////E STEP/////////////////
        {
            // STEP 1
            alpha_samples.push_back(sampler.resample_from(pi_tree, N));
            alpha_trees.push_back(DETree(alpha_samples[0], pi_low_limit, pi_high_limit));

            // STEP 2
            for (size_t t = 1; t <= T; t++){
                // STEP 2(a)
                vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha_samples[t - 1], N);
                double sum_densities = 0.0;

                for (size_t i = 0; i < temp.size(); i++){
                    // STEP 2(b)
                    Sample x = sampler.sample_given(m_tree, temp[i]);

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
                alpha_samples.push_back(temp);
                alpha_trees.push_back(DETree(temp, pi_low_limit, pi_high_limit));
            }

            // STEP 3
            beta_samples.push_back(sampler.uniform_sampling(pi_low_limit, pi_high_limit, N));
            beta_trees.push_back(DETree(beta_samples[0], pi_low_limit, pi_high_limit));

            // STEP 4
            for (size_t t = T; t >= 1; t--){
                // STEP 4(a)
                int index_t = (T + 1) - (t + 1);
                vector<Sample> temp = sampler.likelihood_weighted_resampler(beta_samples[index_t], N);
                double sum_densities = 0.0;

                for (size_t i = 0; i < temp.size(); i++){
                    // STEP 4(b)
                    Sample x = sampler.sample_given(m_tree, temp[i]);

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
                beta_samples.push_back(temp);
                beta_trees.push_back(DETree(temp, pi_low_limit, pi_high_limit));
            }

            // STEP 5
            for (size_t t = 1; t <= T; t++){
                vector<Sample> temp;
                double sum_density = 0.0;

                int index_t = (T + 1) - (t);

                // STEP 5(a)
                for (int j = 0; j < N / 2; j++){
                    Sample sample = sampler.likelihood_weighted_sampler(alpha_samples[t]);
                    sample.p = beta_trees[index_t].density_value(sample, rho);
                    sum_density += sample.p;
                    vector<vector<Sample> > alpha_samples;
                    temp.push_back(sample);
                }

                // STEP 5(b)
                for (int j = 0; j < N - (N / 2); j++){
                    Sample sample = sampler.likelihood_weighted_sampler(beta_samples[index_t]);
                    sample.p = alpha_trees[t].density_value(sample, rho);
                    sum_density += sample.p;
                    temp.push_back(sample);
                }

                // Normalizing the probabilities
                for (size_t i = 0; i < temp.size(); i++){
                    temp[i].p = temp[i].p / sum_density;
                }

                gamma_samples.push_back(temp);
                gamma_trees.push_back(DETree(temp, pi_low_limit, pi_high_limit));
            }
        }

        /////////////////M STEP/////////////////
        {
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine gen(seed);

            vector<Sample> temp_m;
            vector<Sample> temp_v;
            vector<Sample> temp_pi;

            // STEP 1
            for (int i = 0; i < N; i++){
                uniform_real_distribution<double> dist(1, T - 1);
                int t = dist(gen);

                Sample x = sampler.likelihood_weighted_sampler(gamma_samples[t]);
                Sample x_prime = sampler.likelihood_weighted_sampler(gamma_samples[t + 1]);

                Sample temp = x.combine(x_prime.values);
                temp.p = 1.0 / N;
                temp_m.push_back(temp);
            }

            // STEP 2
            for (int i = 0; i < N; i++){
                uniform_real_distribution<double> dist(1, T);
                int t = dist(gen);

                Sample x = sampler.likelihood_weighted_sampler(gamma_samples[t]);

                Sample temp = (*observations)[t].combine(x);
                temp.p = 1.0 / N;
                temp_v.push_back(temp);
            }

            // STEP 3
            for (size_t i = 0; i < gamma_samples[0].size(); i++){
                temp_pi.push_back(gamma_samples[0][i]);
            }

            pi = &temp_pi;
            pi_tree->create_tree(*pi, pi_low_limit, pi_high_limit);

            m =  &temp_m;
            m_tree->create_tree(*m, m_low_limit, m_high_limit);

            v = &temp_v;
            v_tree->create_tree(*v, v_low_limit, v_high_limit);
        }

        /////////////////ANNEALING/////////////////
        rho = rho * rho_bar;

        /////////////////SAMPLE SET SIZE/////////////////
        if (N < (int)max_sample_size)
            N = N * eta;

        /////////////////STOP CONDITION/////////////////
        LOG(INFO) << "Iteration " << iteration + 1 << " Finished!";
        iteration++;
        if (iteration >= max_iteration){
            cond = false;
        }
    }

}

void MCFHMM::set_limits(vector<double> *pi_low_limit, vector<double> *pi_high_limit,
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

void MCFHMM::init_hmm(int sample_size_pi, int sample_size_m, int sample_size_v){

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

}
