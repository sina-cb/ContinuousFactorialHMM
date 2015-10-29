#include "MCFHMM.h"
#include <random>
#include <chrono>
#include <cmath>
using namespace std;

MCFHMM::MCFHMM(){
    pi = new vector<pi_type>();
    m = new vector<m_type>();
    v = new vector<v_type>();
}


void MCFHMM::init_hmm(int sample_size_pi, int sample_size_m, int sample_size_v){

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);

    uniform_real_distribution<double> X_dist(0, 10);
    uniform_real_distribution<double> Y_dist(0, 10);
    uniform_real_distribution<double> TH_dist(0, 2 * M_PI);
    uniform_real_distribution<double> P_dist(0, 1);


    for (int i = 0; i < sample_size_pi; i++){
        pi_type sample;
        sample.push_back(X_dist(gen));
        sample.push_back(Y_dist(gen));
        sample.push_back(TH_dist(gen));
        sample.push_back(P_dist(gen));
        pi->push_back(sample);
    }

    for (int i = 0; i < sample_size_m; i++){
        m_type sample;
        sample.push_back(X_dist(gen));
        sample.push_back(Y_dist(gen));
        sample.push_back(TH_dist(gen));
        sample.push_back(X_dist(gen));
        sample.push_back(Y_dist(gen));
        sample.push_back(TH_dist(gen));
        sample.push_back(P_dist(gen));
        m->push_back(sample);
    }

    for (int i = 0; i < sample_size_v; i++){
        v_type sample;
        sample.push_back(X_dist(gen));
        sample.push_back(Y_dist(gen));
        sample.push_back(TH_dist(gen));
        sample.push_back(X_dist(gen));
        sample.push_back(Y_dist(gen));
        sample.push_back(TH_dist(gen));
        sample.push_back(P_dist(gen));
        v->push_back(sample);
    }

}
