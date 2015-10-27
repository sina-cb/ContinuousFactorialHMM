#include <iostream>
#include <vector>
#include <tuple>
using namespace std;

#ifndef MCFHMM_H
#define MCFHMM_H

typedef tuple<double, double, double> pi_type;
typedef tuple<double, double, double, double, double, double> m_type;
typedef tuple<double, double, double, double, double, double> v_type;

class MCFHMM{

private:

    // Model parameters
    vector<pi_type> *pi;    // Initial State Distribution
    vector<m_type> *m;      // Transition Model
    vector<v_type> *v;      // Observation Model


public:
    MCFHMM();

    void init_hmm(int sample_size_pi, int sample_size_m, int sample_size_v);

    vector<pi_type>* get_pi(){
       return pi;
    }

    void set_pi(vector<pi_type> *pi){
        this->pi = pi;
    }

    vector<m_type>* get_m(){
       return m;
    }

    void set_m(vector<m_type> *m){
        this->m = m;
    }

    vector<v_type>* get_v(){
       return v;
    }

    void set_v(vector<v_type> *v){
        this->v = v;
    }

};

#endif
