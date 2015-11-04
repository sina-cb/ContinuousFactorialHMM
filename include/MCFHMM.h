#include <iostream>
#include <vector>
#include <tuple>
#include "Sample.h"
using namespace std;

#ifndef MCFHMM_H
#define MCFHMM_H

class MCFHMM{

private:

    // Model parameters
    vector<Sample> *pi;    // Initial State Distribution
    vector<Sample> *m;      // Transition Model
    vector<Sample> *v;      // Observation Model


public:
    MCFHMM();

    void init_hmm(int sample_size_pi, int sample_size_m, int sample_size_v);

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
