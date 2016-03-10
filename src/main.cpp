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
#define MAX_ITERATION 3
#define INIT_OBS_C 100
#define TEST_OBS_C 50
#define PI_SAMPLE_C 20
#define M_SAMPLE_C 200
#define V_SAMPLE_C 200

vector<string> hmm_types = {"Monte Carlo HMM", "Layered Monte Carlo HMM"};

void init_GLOG();
void init_limits_hmm_0(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_limits_hmm_1(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_observations(vector<Observation> * obs, size_t size);

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

    LMCHMM hmm(2);
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        hmm.set_limits(pi_low_limits_0, pi_high_limits_0, m_low_limits_0, m_high_limits_0, v_low_limits_0, v_high_limits_0, 0);
        hmm.set_limits(pi_low_limits_1, pi_high_limits_1, m_low_limits_1, m_high_limits_1, v_low_limits_1, v_high_limits_1, 1);
        hmm.learn_hmm(observations, MAX_ITERATION, N);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Generating the MCHMM time: " << (t2 - t1) << " seconds";
    }

    return 0;
}

void init_limits_hmm_0(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    pi_low_limits->push_back(1);

    pi_high_limits->push_back(2);

    m_low_limits->push_back(1);
    m_low_limits->push_back(1);

    m_high_limits->push_back(2);
    m_high_limits->push_back(2);

    v_low_limits->push_back(0);
    v_low_limits->push_back(0);
    v_low_limits->push_back(0);

    v_high_limits->push_back(6);
    v_high_limits->push_back(6);
    v_high_limits->push_back(6);

}

void init_limits_hmm_1(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    pi_low_limits->push_back(1);

    pi_high_limits->push_back(2);

    m_low_limits->push_back(1);
    m_low_limits->push_back(1);

    m_high_limits->push_back(2);
    m_high_limits->push_back(2);

    v_low_limits->push_back(1);
    v_low_limits->push_back(1);

    v_high_limits->push_back(2);
    v_high_limits->push_back(2);

}

void init_observations(vector<Observation> *obs, size_t size){

    if (!obs){
        LOG(FATAL) << "Observation vector was NULL!";
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(-.1, .1);
    double x = 5, y = 5;
    for (size_t i = 0; i < size / 2; i++){
        Observation temp1;
        temp1.values.push_back(.1 + dist(gen));
        temp1.values.push_back(.1 + dist(gen));
        obs->push_back(temp1);

        Observation temp2;
        temp2.values.push_back(x + dist(gen));
        temp2.values.push_back(y + dist(gen));
        obs->push_back(temp2);
    }

    if (size % 2 == 1){
        Observation temp1;
        temp1.values.push_back(.1 + dist(gen));
        temp1.values.push_back(.1 + dist(gen));
        obs->push_back(temp1);
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
