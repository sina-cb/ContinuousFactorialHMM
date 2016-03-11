#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <glog/logging.h>
#include "Sampler.h"
#include "Sample.h"
#include "MCHMM.h"
#include "LMCHMM.h"
#include "DETree.h"
#include "Observation.h"
#include "Timer.h"
#include <fstream>
#include <sstream>
using namespace std;
using namespace google;

#define HMM_TYPE 1

#define N 100
#define N_HMM_TEST 100
#define MAX_ITERATION 2
#define INIT_OBS_C 100
#define TEST_OBS_C 50
#define PI_SAMPLE_C 20
#define M_SAMPLE_C 200
#define V_SAMPLE_C 200

#define USE_EM_ONLY 0

vector<string> hmm_types = {"Monte Carlo HMM", "Layered Monte Carlo HMM"};

void init_GLOG();
void init_limits_hmm_0(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_limits_hmm_1(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_observations(vector<Observation> * obs, size_t size);
void init_distributions(vector<Observation> * obs,
                        vector<Sample> * pi_0, vector<Sample> * m_0, vector<Sample> * v_0,
                        vector<Sample> * pi_1, vector<Sample> * m_1, vector<Sample> * v_1);

int main(int argc, char** argv){
    init_GLOG();
    LOG(INFO) << "Using \"" << hmm_types[HMM_TYPE] << "\" algorithm.";

    vector<double> * pi_low_limits_0 = new vector<double>();
    vector<double> * pi_high_limits_0 = new vector<double>();
    vector<double> * m_low_limits_0 = new vector<double>();
    vector<double> * m_high_limits_0 = new vector<double>();
    vector<double> * v_low_limits_0 = new vector<double>();
    vector<double> * v_high_limits_0 = new vector<double>();

    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_limits_hmm_0(pi_low_limits_0, pi_high_limits_0, m_low_limits_0,
                          m_high_limits_0, v_low_limits_0, v_high_limits_0);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the HMM_0 limits time: " << (t2 - t1) << " seconds";
    }

    vector<double> * pi_low_limits_1 = new vector<double>();
    vector<double> * pi_high_limits_1 = new vector<double>();
    vector<double> * m_low_limits_1 = new vector<double>();
    vector<double> * m_high_limits_1 = new vector<double>();
    vector<double> * v_low_limits_1 = new vector<double>();
    vector<double> * v_high_limits_1 = new vector<double>();

    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_limits_hmm_1(pi_low_limits_1, pi_high_limits_1, m_low_limits_1,
                          m_high_limits_1, v_low_limits_1, v_high_limits_1);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the HMM_1 limits time: " << (t2 - t1) << " seconds";
    }

    vector<Observation> *observations = new vector<Observation>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_observations(observations, INIT_OBS_C);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the observations time: " << (t2 - t1) << " seconds";
    }

    vector<Sample> * pi_0 = new vector<Sample>();
    vector<Sample> *  m_0 = new vector<Sample>();
    vector<Sample> *  v_0 = new vector<Sample>();

    vector<Sample> * pi_1 = new vector<Sample>();
    vector<Sample> *  m_1 = new vector<Sample>();
    vector<Sample> *  v_1 = new vector<Sample>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_distributions(observations, pi_0, m_0, v_0, pi_1, m_1, v_1);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the distributions time: " << (t2 - t1) << " seconds";
    }

    LMCHMM hmm(2);
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        hmm.set_limits(pi_low_limits_0, pi_high_limits_0, m_low_limits_0, m_high_limits_0, v_low_limits_0, v_high_limits_0, 0);
        hmm.set_limits(pi_low_limits_1, pi_high_limits_1, m_low_limits_1, m_high_limits_1, v_low_limits_1, v_high_limits_1, 1);

#if !USE_EM_ONLY
        hmm.set_distributions(pi_0, m_0, v_0, 0.5, 0);
        hmm.set_distributions(pi_1, m_1, v_1, 0.5, 1);
#endif

        hmm.learn_hmm(observations, MAX_ITERATION, N);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Generating the MCHMM time: " << (t2 - t1) << " seconds";
    }

    // Testing
    {
        Sampler sampler;

        for (size_t i = 0; i < observations->size() - 4; i++){
            vector<Observation> obs;
            for (size_t j = 0; j <= i; j++){
                obs.push_back((*observations)[j]);
            }
            vector<DETree *> trees = hmm.forward(&obs, N);

            Sample sample_0 = sampler.sample(trees[0]);
            Sample sample_1 = sampler.sample(trees[1]);

            LOG(INFO) << "Sample V: " << sample_0.values[0] << "\tSample A: " << sample_1.values[0];

            for (size_t i = 0; i < trees.size(); i++){
                delete trees[i];
            }
        }

    }

    return 0;
}

double min_x = 0;
double max_x = 10;

double min_v = -5;
double max_v = 5;

double min_a = -1;
double max_a = 1;

void init_limits_hmm_0(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    pi_low_limits->push_back(min_v);

    pi_high_limits->push_back(max_v);

    m_low_limits->push_back(min_v);
    m_low_limits->push_back(min_v);

    m_high_limits->push_back(max_v);
    m_high_limits->push_back(max_v);

    v_low_limits->push_back(min_x);
    v_low_limits->push_back(min_v);

    v_high_limits->push_back(max_x);
    v_high_limits->push_back(max_v);
}

void init_limits_hmm_1(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    pi_low_limits->push_back(min_a);

    pi_high_limits->push_back(max_a);

    m_low_limits->push_back(min_a);
    m_low_limits->push_back(min_a);

    m_high_limits->push_back(max_a);
    m_high_limits->push_back(max_a);

    v_low_limits->push_back(min_v);
    v_low_limits->push_back(min_a);

    v_high_limits->push_back(max_v);
    v_high_limits->push_back(max_a);
}

void init_observations(vector<Observation> *obs, size_t size){

    if (!obs){
        LOG(FATAL) << "Observation vector was NULL!";
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(-.1, .1);
    Observation temp;
    temp.values.push_back(0.0);
    obs->push_back(temp);
    for (size_t i = 0; i < size / 2; i++){
        Observation temp1;
        temp1.values.push_back(obs->back().values[0] + ((max_x - 0.5) / (size / 2)) + dist(gen));
        obs->push_back(temp1);
    }

    for (size_t i = 0; i < size - (size / 2) - 1; i++){
        Observation temp1;
        temp1.values.push_back(obs->back().values[0] - ((max_x - 0.5) / (size / 2)) + dist(gen));
        if (temp1.values[0] < 0){
            temp1.values[0] = 0.0;
        }
        obs->push_back(temp1);
    }
}

void init_distributions(vector<Observation> *obs,
                        vector<Sample> *pi_0, vector<Sample> *m_0, vector<Sample> *v_0,
                        vector<Sample> *pi_1, vector<Sample> *m_1, vector<Sample> *v_1){

    LOG(INFO) << "Let's initialize the distributions based on the observation and physic laws!";
    LOG(INFO) << "I assume that time steps == 1s";

    vector<Sample> vels;
    for (size_t i = 0; i < obs->size() - 1; i++){
        Sample vel;
        vel.values.push_back((*obs)[i + 1].values[0] - (*obs)[i].values[0]);
        vel.p = 1.0 / (obs->size() - 1);
        vels.push_back(vel);
    }

    vector<Sample> accs;
    for (size_t i = 0; i < vels.size() - 1; i++){
        Sample acc;
        acc.values.push_back((vels)[i + 1].values[0] - (vels)[i].values[0]);
        acc.p = 1.0 / (vels.size() - 1);
        accs.push_back(acc);
    }

    for (size_t i = 0; i < 20; i++){
        Sample vel;
        vel.values.push_back(vels[0].values[0]);
        pi_0->push_back(vels[0]);
        pi_1->push_back(accs[0]);
    }

    for (size_t i = 0; i < vels.size() - 1; i++){
        Sample m_s;
        m_s.values.push_back(vels[i + 1].values[0]);
        m_s.values.push_back(vels[i].values[0]);
        m_s.p = 1.0 / (vels.size() - 1);
        m_0->push_back(m_s);
    }

    for (size_t i = 0; i < vels.size(); i++){
        Sample v_s;
        v_s.values.push_back((*obs)[i + 1].values[0]);
        v_s.values.push_back(vels[i].values[0]);
        v_s.p = 1.0 / (vels.size());
        v_0->push_back(v_s);
    }

    for (size_t i = 0; i < accs.size() - 1; i++){
        Sample m_s;
        m_s.values.push_back(accs[i + 1].values[0]);
        m_s.values.push_back(accs[i].values[0]);
        m_s.p = 1.0 / (accs.size() - 1);
        m_1->push_back(m_s);
    }

    for (size_t i = 0; i < accs.size(); i++){
        Sample v_s;
        v_s.values.push_back(vels[i + 1].values[0]);
        v_s.values.push_back(accs[i].values[0]);
        v_s.p = 1.0 / (accs.size());
        v_1->push_back(v_s);
    }

}

void init_GLOG(){
    InitGoogleLogging(((string)hmm_types[HMM_TYPE]).c_str());
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}
