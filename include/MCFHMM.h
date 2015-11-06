#include <iostream>
#include <vector>
#include <tuple>
#include "Sample.h"
#include "Observation.h"
using namespace std;

#ifndef MCFHMM_H
#define MCFHMM_H

class MCFHMM{

private:

    // Model parameters
    vector<Sample> *pi;    // Initial State Distribution
    vector<Sample> *m;      // Transition Model
    vector<Sample> *v;      // Observation Model

    vector<double> *pi_low_limit = NULL;
    vector<double> *pi_high_limit = NULL;

    vector<double> *m_low_limit = NULL;
    vector<double> *m_high_limit = NULL;

    vector<double> *v_low_limit = NULL;
    vector<double> *v_high_limit = NULL;


public:
    MCFHMM();

    void init_hmm(int sample_size_pi, int sample_size_m, int sample_size_v);
    void set_limits(vector<double> *pi_low_limit, vector<double> *pi_high_limit,
                    vector<double> *m_low_limit, vector<double> *m_high_limit,
                    vector<double> *v_low_limit, vector<double> *v_high_limit
                    );
    void learn_hmm(vector<Observation> *observations, int max_iteration, int N);

    vector<Sample>* get_pi(){
       return pi;
    }

    void set_pi(vector<Sample> *pi){
        this->pi = pi;
    }

    vector<Sample>* get_m(){
       return m;
    }

    void set_m(vector<Sample> *m){
        this->m = m;
    }

    vector<Sample>* get_v(){
       return v;
    }

    void set_v(vector<Sample> *v){
        this->v = v;
    }

};

#endif
