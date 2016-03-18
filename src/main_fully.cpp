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

#define N 30
//#define N_HMM_TEST 100
#define MAX_ITERATION 10
#define INIT_OBS_C 500
//#define TEST_OBS_C 50init_limits_hmm_1
//#define PI_SAMPLE_C 20
//#define M_SAMPLE_C 200
//#define V_SAMPLE_C 200

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
void init_distributions(vector<Sample> * pi_0, vector<Sample> * m_0, vector<Sample> * v_0,
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

        init_distributions(pi_0, m_0, v_0, pi_1, m_1, v_1);

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

        hmm.learn_hmm_separately(observations, MAX_ITERATION, N);

        //hmm.learn_hmm(observations, MAX_ITERATION, N);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Generating the MCHMM time: " << (t2 - t1) << " seconds";
    }

    // Testing
    {
        Sampler sampler;
        int true_vel = 0;

        size_t count_ = (observations->size() - 4);
        for (size_t i = 0; i < count_; i++){
            vector<Observation> obs;
            for (size_t j = 0; j <= i; j++){
                obs.push_back((*observations)[j]);
            }
            vector<DETree *> trees = hmm.forward(&obs, N);

            Sample sample_0 = sampler.sample_avg(trees[0], 5);
            Sample sample_1 = sampler.sample_avg(trees[1], 5);

            LOG(INFO) << setprecision(3) << i << ":\tSample V: " << sample_0.values[0] << "\t" << sample_0.values[1];
            LOG(INFO) << setprecision(3) << i << ":\tSample A: " << sample_1.values[0] << "\t" << sample_1.values[1];
            LOG(INFO) << "";

            for (size_t i = 0; i < trees.size(); i++){
                delete trees[i];
            }
        }

//        LOG(INFO) << "Trues: " << true_vel << "\tAccuracy: " << (((double)true_vel) / count_ * 100) << "%";
    }

    return 0;
}

double min_v = -0.4;
double max_v = 0.4;

double min_a = -.1;
double max_a = .1;

double min_cross = 0;
double max_cross = 1;

double min_junc = 0;
double max_junc = 2;

double min_wall = 0;
double max_wall = 3;

void init_limits_hmm_0(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    pi_low_limits->push_back(min_v);
    pi_low_limits->push_back(min_v);

    pi_high_limits->push_back(max_v);
    pi_high_limits->push_back(max_v);

    m_low_limits->push_back(min_v);
    m_low_limits->push_back(min_v);
    m_low_limits->push_back(min_v);
    m_low_limits->push_back(min_v);

    m_high_limits->push_back(max_v);
    m_high_limits->push_back(max_v);
    m_high_limits->push_back(max_v);
    m_high_limits->push_back(max_v);

    v_low_limits->push_back(min_cross);
    v_low_limits->push_back(min_junc);
    v_low_limits->push_back(min_wall);
    v_low_limits->push_back(min_wall);
    v_low_limits->push_back(min_v);
    v_low_limits->push_back(min_v);

    v_high_limits->push_back(max_cross);
    v_high_limits->push_back(max_junc);
    v_high_limits->push_back(max_wall);
    v_high_limits->push_back(max_wall);
    v_high_limits->push_back(max_v);
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
    pi_low_limits->push_back(min_a);

    pi_high_limits->push_back(max_a);
    pi_high_limits->push_back(max_a);

    m_low_limits->push_back(min_a);
    m_low_limits->push_back(min_a);
    m_low_limits->push_back(min_a);
    m_low_limits->push_back(min_a);

    m_high_limits->push_back(max_a);
    m_high_limits->push_back(max_a);
    m_high_limits->push_back(max_a);
    m_high_limits->push_back(max_a);

    v_low_limits->push_back(min_v);
    v_low_limits->push_back(min_v);
    v_low_limits->push_back(min_a);
    v_low_limits->push_back(min_a);

    v_high_limits->push_back(max_v);
    v_high_limits->push_back(max_v);
    v_high_limits->push_back(max_a);
    v_high_limits->push_back(max_a);
}

void init_observations(vector<Observation> *obs, size_t size){

    if (!obs){
        LOG(FATAL) << "Observation vector was NULL!";
    }

    ifstream file("/home/sina/Desktop/obs.txt");

    if (file.is_open()){
        string line;
        getline(file, line); // Skip the header

        while(getline(file, line)){
            stringstream ssin(line);
            Observation sample_obs;

            while(ssin.good()){
                string temp;
                ssin >> temp;
                sample_obs.values.push_back(atof(temp.c_str()));
            }

            obs->push_back(sample_obs);
        }

    }else{
        LOG(FATAL) << "Cannot open Obs input file!";
    }

    file.close();

}

void init_distributions(vector<Sample> *pi_0, vector<Sample> *m_0, vector<Sample> *v_0,
                        vector<Sample> *pi_1, vector<Sample> *m_1, vector<Sample> *v_1){

    for (size_t i = 0; i < 20; i++){
        Sample zero_vel;
        zero_vel.values.push_back(0);
        zero_vel.values.push_back(0);
        zero_vel.p = 1.0 / 20.0;

        Sample zero_accl;
        zero_accl.values.push_back(0);
        zero_accl.values.push_back(0);
        zero_accl.p = 1.0 / 20.0;

        pi_0->push_back(zero_vel);
        pi_1->push_back(zero_accl);
    }

    if (!m_0){
        LOG(FATAL) << "M_0 vector was not initialized!";
    }

    ifstream file("/home/sina/Desktop/m_1.txt");

    if (file.is_open()){
        string line;
        getline(file, line); // Skip the headers

        while(getline(file, line)){
            stringstream ssin(line);
            Sample sample_m;

            while(ssin.good()){
                string temp;
                ssin >> temp;
                sample_m.values.push_back(atof(temp.c_str()));
            }
            sample_m.values.pop_back();

            m_0->push_back(sample_m);
        }

    }else{
        LOG(FATAL) << "Cannot open M_1 input file!";
    }

    file.close();

    if (!m_1){
        LOG(FATAL) << "M_1 vector was not initialized!";
    }

    file.open("/home/sina/Desktop/m_2.txt");

    if (file.is_open()){
        string line;
        getline(file, line); // Skip the headers

        while(getline(file, line)){
            stringstream ssin(line);
            Sample sample_m;

            while(ssin.good()){
                string temp;
                ssin >> temp;
                sample_m.values.push_back(atof(temp.c_str()));
            }
            sample_m.values.pop_back();

            m_1->push_back(sample_m);
        }

    }else{
        LOG(FATAL) << "Cannot open M_2 input file!";
    }

    file.close();


    if (!v_0){
        LOG(FATAL) << "V_0 vector was not initialized!";
    }

    file.open("/home/sina/Desktop/v_1.txt");

    if (file.is_open()){
        string line;
        getline(file, line); // Skip the headers

        while(getline(file, line)){
            stringstream ssin(line);
            Sample sample_v;

            while(ssin.good()){
                string temp;
                ssin >> temp;
                sample_v.values.push_back(atof(temp.c_str()));
            }
            sample_v.values.pop_back();

            v_0->push_back(sample_v);
        }

    }else{
        LOG(FATAL) << "Cannot open V_1 input file!";
    }

    file.close();

    if (!v_1){
        LOG(FATAL) << "V_1 vector was not initialized!";
    }

    file.open("/home/sina/Desktop/v_2.txt");

    if (file.is_open()){
        string line;
        getline(file, line); // Skip the headers

        while(getline(file, line)){
            stringstream ssin(line);
            Sample sample_v;

            while(ssin.good()){
                string temp;
                ssin >> temp;
                sample_v.values.push_back(atof(temp.c_str()));
            }
            sample_v.values.pop_back();

            v_1->push_back(sample_v);
        }

    }else{
        LOG(FATAL) << "Cannot open V_2 input file!";
    }

    file.close();
}

void init_GLOG(){
    InitGoogleLogging(((string)hmm_types[HMM_TYPE]).c_str());
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}
