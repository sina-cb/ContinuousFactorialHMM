#include "MCFHMM.h"
#include <cmath>
#include "DETree.h"
#include "Sampler.h"
using namespace std;

MCFHMM::MCFHMM(){
    pi = new vector<Sample>();
    m = new vector<Sample>();
    v = new vector<Sample>();
}

void MCFHMM::learn_hmm(vector<Observation> *observations, int max_iteration, int N){

    Sampler sampler;

    DETree pi_tree;
    DETree m_tree;
    DETree v_tree;

    pi_tree.create_tree(*pi, pi_low_limit, pi_high_limit);
    m_tree.create_tree(*m, m_low_limit, m_high_limit);
    v_tree.create_tree(*v, v_low_limit, v_high_limit);

    for (size_t i = 0; i < 100; i++){
        Sample temp = sampler.sample(&pi_tree);
        LOG(INFO) << "SAMPLE: " << temp.str();
        LOG(INFO) << "Probability: " << pi_tree.density_value(temp);
    }

    cout << "First \n\n";
    for (size_t i = 0; i < pi->size(); i++){
        for (size_t j = 0; j < (*pi)[i].values.size(); j++){
            cout << (*pi)[i].values[j] << " ";
        }
        cout << endl;
    }
    pi = sampler.likelihood_weighted_sampler(*pi);
    cout << "Then \n\n";
    for (size_t i = 0; i < pi->size(); i++){
        for (size_t j = 0; j < (*pi)[i].values.size(); j++){
            cout << (*pi)[i].values[j] << " ";
        }
        cout << endl;
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

}
