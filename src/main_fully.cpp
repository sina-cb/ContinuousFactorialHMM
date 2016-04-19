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

#define N 40
//#define N_HMM_TEST 100
#define MAX_ITERATION 100
//#define TEST_OBS_C 50init_limits_hmm_1
//#define PI_SAMPLE_C 20
//#define M_SAMPLE_C 200
//#define V_SAMPLE_C 200

#define USE_EM_ONLY 0
#define GRID_SEARCH_CV 0

vector<string> hmm_types = {"Monte Carlo HMM", "Layered Monte Carlo HMM"};

void init_GLOG();
void init_limits_hmm_0(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_limits_hmm_1(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_observations(vector<Observation> * obs);
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

        init_observations(observations);

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

#if !GRID_SEARCH_CV

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

        hmm.learn_hmm_separately_KL(observations, 0.4, MAX_ITERATION, N);

        //hmm.learn_hmm_KL(observations, 0.4, MAX_ITERATION, N);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Generating the MCHMM time: " << (t2 - t1) << " seconds";
    }

    // Testing
    {
        Sampler sampler;

        size_t count_ = (observations->size() - 4);
        size_t true_count = 0;
        size_t true_count_a = 0;
        for (size_t i = 0; i < count_; i++){
            vector<Observation> obs;
            for (size_t j = 0; j <= i; j++){
                obs.push_back((*observations)[j]);
            }
            vector<DETree *> trees = hmm.forward(&obs, N);

            Sample sample_0 = sampler.sample_avg(trees[0], 5);
            Sample sample_1 = sampler.sample_avg(trees[1], 5);

            double est = std::sqrt(pow(sample_0.values[0], 2) + pow(sample_0.values[1], 2));
            double real1 = std::sqrt(pow((*m_0)[i].values[0], 2) + pow((*m_0)[i].values[1], 2));
            double real2 = std::sqrt(pow((*m_0)[i].values[2], 2) + pow((*m_0)[i].values[3], 2));

            double est_a = std::sqrt(pow(sample_1.values[0], 2) + pow(sample_1.values[1], 2));
            double real1_a = std::sqrt(pow((*m_1)[i].values[0], 2) + pow((*m_1)[i].values[1], 2));
            double real2_a = std::sqrt(pow((*m_1)[i].values[2], 2) + pow((*m_1)[i].values[3], 2));

            if (std::abs(est - real1) < 0.1 || std::abs(est - real2) < 0.1){
                true_count++;
            }

            if (std::abs(est_a - real1_a) < 0.05 || std::abs(est_a - real2_a) < 0.05){
                true_count_a++;
            }

            LOG(INFO) << setprecision(3) << i << ":\tSample V: " << est;
            LOG(INFO) << setprecision(3) << i << ":\tReal V:   " << real1;
            LOG(INFO) << setprecision(3) << i << ":\tReal V:   " << real2;
            //            LOG(INFO) << setprecision(3) << i << ":\tSample A: " << sample_1.values[0] << "\t" << sample_1.values[1];
            LOG(INFO) << "";
            LOG(INFO) << setprecision(3) << i << ":\tSample a: " << est_a;
            LOG(INFO) << setprecision(3) << i << ":\tReal a:   " << real1_a;
            LOG(INFO) << setprecision(3) << i << ":\tReal a:   " << real2_a;
            LOG(INFO) << "";

            for (size_t i = 0; i < trees.size(); i++){
                delete trees[i];
            }
        }

        LOG(INFO) << "Trues: " << true_count << "\tAccuracy: " << (((double)true_count) / count_ * 100) << "%";
        LOG(INFO) << "Trues a: " << true_count_a << "\tAccuracy a: " << (((double)true_count_a) / count_ * 100) << "%";
    }
#else

    vector<int> N_values;
    vector<int> max_iteration_values;

    for (size_t i = 50; i <= 200; i += 50){
        N_values.push_back(i);
    }

    for (size_t i = 2; i <= 10; i += 2){
        max_iteration_values.push_back(i);
    }

    vector<string> i_and_js;
    vector<double> accuracies;
    vector<double> times;

    for (size_t i = 0; i < N_values.size(); i++){
        for (size_t j = 0; j < max_iteration_values.size(); j++){
            int N_ = N_values[i];
            int max_iteration_ = max_iteration_values[j];

            stringstream sst;
            sst << "N = " << N_ << ", MAX_ITERATION = " << max_iteration_;
            i_and_js.push_back(sst.str());

            LOG(WARNING) << "Testing " << sst.str();

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

                //hmm.learn_hmm_separately(observations, MAX_ITERATION, N);

                hmm.learn_hmm(observations, max_iteration_, N_);

                double t2 = tmr.elapsed();
                LOG(INFO) << "Generating the MCHMM time: " << (t2 - t1) << " seconds";
                times.push_back(t2 - t1);
            }

            // Testing
            {
                Sampler sampler;

                size_t count_ = (observations->size() - 4);
                size_t true_count = 0;
                for (size_t i = 0; i < count_; i++){
                    vector<Observation> obs;
                    for (size_t j = 0; j <= i; j++){
                        obs.push_back((*observations)[j]);
                    }
                    vector<DETree *> trees = hmm.forward(&obs, N_);

                    Sample sample_0 = sampler.sample_avg(trees[0], 5);
                    Sample sample_1 = sampler.sample_avg(trees[1], 5);

                    double est = std::sqrt(pow(sample_0.values[0], 2) + pow(sample_0.values[1], 2));
                    double real1 = std::sqrt(pow((*m_0)[i].values[0], 2) + pow((*m_0)[i].values[1], 2));
                    double real2 = std::sqrt(pow((*m_0)[i].values[2], 2) + pow((*m_0)[i].values[3], 2));

                    if (std::abs(est - real1) < 0.1 || std::abs(est - real2) < 0.1){
                        true_count++;
                    }

                    LOG(INFO) << setprecision(3) << i << ":\tSample V: " << est;
                    LOG(INFO) << setprecision(3) << i << ":\tReal V:   " << real1;
                    LOG(INFO) << setprecision(3) << i << ":\tReal V:   " << real2;
                    //            LOG(INFO) << setprecision(3) << i << ":\tSample A: " << sample_1.values[0] << "\t" << sample_1.values[1];
                    LOG(INFO) << "";

                    for (size_t i = 0; i < trees.size(); i++){
                        delete trees[i];
                    }
                }

                LOG(INFO) << "Trues: " << true_count << "\tAccuracy: " << (((double)true_count) / count_ * 100) << "%";
                accuracies.push_back((((double)true_count) / count_ * 100));
            }
        }
    }

    // Print results from the Grid Search
    {
        for (size_t i = 0; i < times.size(); i++){
            LOG(WARNING) << i_and_js[i] << "\t" << accuracies[i] << "\t" << times[i];
        }
    }

#endif

    return 0;
}

double min_v = -0.8;
double max_v = 0.8;

double min_a = -0.5;
double max_a = +0.5;

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

void init_observations(vector<Observation> *obs){

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
    FLAGS_minloglevel = 1;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}
