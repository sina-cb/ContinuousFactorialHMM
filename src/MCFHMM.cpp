#include "MCFHMM.h"
#include <cmath>
#include "Sampler.h"
using namespace std;

MCFHMM::MCFHMM(){
    pi = new vector<Sample>();
    m = new vector<Sample>();
    v = new vector<Sample>();
}

void MCFHMM::learn_hmm(vector<Observation> *observations, int max_iteration, int N){
    Sampler sampler;

    init_hmm(N, N, N, 0.2);

    vector<vector<Sample> > alpha;
    vector<vector<Sample> > beta;
    vector<vector<Sample> > gamma;

    /////////////////E STEP/////////////////
    {
        alpha.push_back(*pi);

        /////////////////Compute Alpha/////////////////
        for (size_t t = 1; t < observations->size(); t++){

            vector<Sample> temp = sampler.likelihood_weighted_resampler(alpha[t - 1], N);
            double sum_densities = 0.0;
            for (size_t i = 0; i < temp.size(); i++){
                Sample x = sampler.sample_given(m_tree, temp[i]);

                Sample v_temp = (*observations)[t].combine(x);
                double density = v_tree->density_value(v_temp, rho);

                sum_densities += density;
                temp[i] = x;
            }

            for (size_t i = 0; i < temp.size(); i++){
                temp[i].p = temp[i].p / sum_densities;
            }


            break;
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

void MCFHMM::init_hmm(int sample_size_pi, int sample_size_m, int sample_size_v, double rho_init){

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

    rho = rho_init;

    pi_tree = new DETree(*pi, pi_low_limit, pi_high_limit);
    m_tree = new DETree(*m, m_low_limit, m_high_limit);
    v_tree = new DETree(*v, v_low_limit, v_high_limit);

}
