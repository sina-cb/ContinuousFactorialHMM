#include <iostream>
#include <vector>
#include <tuple>
#include "Sample.h"
#include "Observation.h"
#include "DETree.h"
#include "MCHMM.h"
using namespace std;

#ifndef LMCHMM_H
#define LMCHMM_H

class LMCHMM{

private:
    vector<MCHMM *> layers;
    vector<bool> ind_initialized;
    double rho;

public:
    LMCHMM();
    LMCHMM(int layers_count);

    void set_hmm_randomly(int sample_size_pi, int sample_size_m, int sample_size_v, size_t level);
    void set_distributions(vector<Sample> * pi, vector<Sample> * m, vector<Sample> * v, double rho, size_t level);
    void set_limits(vector<double> *pi_low_limit, vector<double> *pi_high_limit,
                    vector<double> *m_low_limit, vector<double> *m_high_limit,
                    vector<double> *v_low_limit, vector<double> *v_high_limit,
                    size_t level);
    void set_layers(size_t layers_count);

    void learn_hmm(vector<Observation> *observations, size_t max_iteration, int N); //TODO: Implement
    vector<DETree> forward(vector<Observation> *observations, size_t N); //TODO: Implement
    vector<Sample> most_probable_seq(); //TODO: Implement

    double _rho();
    bool initialized_();
    size_t _layers_count();
};

#endif
