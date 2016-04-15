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

#define USE_EM_ONLY 0

vector<string> hmm_types = {"Monte Carlo HMM", "Layered Monte Carlo HMM"};

void init_GLOG();
void init_limits_hmm_0(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_limits_hmm_1(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                       vector<double> * m_low_limits, vector<double> * m_high_limits,
                       vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_distributions(vector<Observation> * obs,
                        vector<Sample> * pi_0, vector<Sample> * m_0, vector<Sample> * v_0,
                        vector<Sample> * pi_1, vector<Sample> * m_1, vector<Sample> * v_1);

size_t N = 100;
size_t MAX_ITERATION = 300;
size_t OBSERVATION_COUNT = 50;
double THRESHOLD = 0.5;
string run_number = "";
size_t number_of_runs = 1;

int main(int argc, char** argv){
    init_GLOG();
    LOG(INFO) << "Using \"" << hmm_types[HMM_TYPE] << "\" algorithm.";
    LOG(INFO) << "Testing two number generators' domain";

    if (argc == 7){
        LOG(WARNING) << "Using passed arguments";
        N = (size_t) atoi(argv[1]);
        MAX_ITERATION = (size_t) atoi(argv[2]);
        OBSERVATION_COUNT = (size_t) atoi(argv[3]);
        THRESHOLD = (double) atof(argv[4]);
        run_number = argv[5];
        number_of_runs = (size_t) atoi(argv[6]);

        stringstream ssd;
        ssd << "N: " << N << "\tMAX_ITERATION: " << MAX_ITERATION << "\tOBSERVATION_COUNT: " << OBSERVATION_COUNT
            << "\tTHRESHOLD: " << THRESHOLD << "\trun_number: " << run_number << "\tnumber_of_runs: " << number_of_runs;
        LOG(WARNING) << ssd.str();

    }else{
        LOG(WARNING) << "Using default arguments";
        LOG(WARNING) << "\n\tIf you want to use the arguments, you should pass the following parameters:\n"
                        "\t\tN(size_t) MAX_ITERATION(size_t) OBSERVATION_COUNT(size_t) THRESHOLD(double) run_number(String) "
                        "number_of_runs(size_t)\n\n";

        stringstream ssd;
        ssd << "N: " << N << "\tMAX_ITERATION: " << MAX_ITERATION << "\tOBSERVATION_COUNT: " << OBSERVATION_COUNT
            << "\tTHRESHOLD: " << THRESHOLD << "\trun_number: " << run_number << "\tnumber_of_runs: " << number_of_runs;
        LOG(WARNING) << ssd.str();
    }

    for (size_t n = 0; n < number_of_runs; n++){

        vector<double> * pi_low_limits_0  = new vector<double>();
        vector<double> * pi_high_limits_0 = new vector<double>();
        vector<double> * m_low_limits_0   = new vector<double>();
        vector<double> * m_high_limits_0  = new vector<double>();
        vector<double> * v_low_limits_0   = new vector<double>();
        vector<double> * v_high_limits_0  = new vector<double>();

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

            //hmm.learn_hmm_separately(observations, MAX_ITERATION, N);

            size_t converged_at = hmm.learn_hmm_KL(observations, THRESHOLD, MAX_ITERATION, N);

            double t2 = tmr.elapsed();
            LOG(WARNING) << "Generating the MCHMM time: " << (t2 - t1) << " seconds";
            LOG(INFO) << "Converged at iteration: " << converged_at;
        }
    }

    // Testing
    //    {
    //        Sampler sampler;
    //        int true_vel = 0;

    //        size_t count_ = (observations->size() - 4) / 2;
    //        double vel_acc = 0.1;
    //        for (size_t i = 0; i < count_; i++){
    //            vector<Observation> obs;
    //            for (size_t j = 0; j <= i; j++){
    //                obs.push_back((*observations)[j]);
    //            }
    //            vector<DETree *> trees = hmm.forward(&obs, N);

    //            Sample sample_0 = sampler.sample_avg(trees[0], 5);
    //            Sample sample_1 = sampler.sample_avg(trees[1], 5);

    //            LOG(INFO) << i << ":\tSample V: " << sample_0.values[0] << "\tSample A: " << sample_1.values[0];

    //            if (std::abs(sample_0.values[0] - (*vels)[i].values[0]) < vel_acc){
    //                true_vel++;
    //            }

    //            for (size_t i = 0; i < trees.size(); i++){
    //                delete trees[i];
    //            }
    //        }

    //        LOG(INFO) << "Trues: " << true_vel << "\tAccuracy: " << (((double)true_vel) / count_ * 100) << "%";
    //    }

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

    double vel_min = -0.5;

    double vel_max = 0.5;

    double crosswalk_min = 0;
    double crosswalk_max = 3;

    double junction_min = 0;
    double junction_max = 3;

    double wall_min = 0;
    double wall_max = 3.0;

    ////////// INIT PI BOUNDS //////////
    pi_low_limits->push_back(vel_min);
    pi_low_limits->push_back(vel_min);

    pi_high_limits->push_back(vel_max);
    pi_high_limits->push_back(vel_max);


    ////////// INIT M  BOUNDS //////////
    m_low_limits->push_back(vel_min);
    m_low_limits->push_back(vel_min);
    m_low_limits->push_back(vel_min);
    m_low_limits->push_back(vel_min);

    m_high_limits->push_back(vel_max);
    m_high_limits->push_back(vel_max);
    m_high_limits->push_back(vel_max);
    m_high_limits->push_back(vel_max);


    ////////// INIT V  BOUNDS //////////
    v_low_limits->push_back(crosswalk_min);
    v_low_limits->push_back(junction_min);
    v_low_limits->push_back(wall_min);
    v_low_limits->push_back(wall_min);
    v_low_limits->push_back(vel_min);
    v_low_limits->push_back(vel_min);

    v_high_limits->push_back(crosswalk_max);
    v_high_limits->push_back(junction_max);
    v_high_limits->push_back(wall_max);
    v_high_limits->push_back(wall_max);
    v_high_limits->push_back(vel_max);
    v_high_limits->push_back(vel_max);
}

void init_limits_hmm_1(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    double accl_min = -0.2;
    double accl_max = 0.2;

    double vel_min = -0.5;
    double vel_max = 0.5;

    ////////// INIT PI BOUNDS //////////
    pi_low_limits->push_back(accl_min);
    pi_low_limits->push_back(accl_min);

    pi_high_limits->push_back(accl_max);
    pi_high_limits->push_back(accl_max);


    ////////// INIT M  BOUNDS //////////
    m_low_limits->push_back(accl_min);
    m_low_limits->push_back(accl_min);
    m_low_limits->push_back(accl_min);
    m_low_limits->push_back(accl_min);

    m_high_limits->push_back(accl_max);
    m_high_limits->push_back(accl_max);
    m_high_limits->push_back(accl_max);
    m_high_limits->push_back(accl_max);


    ////////// INIT V  BOUNDS //////////
    v_low_limits->push_back(vel_min);
    v_low_limits->push_back(vel_min);
    v_low_limits->push_back(accl_min);
    v_low_limits->push_back(accl_min);

    v_high_limits->push_back(vel_max);
    v_high_limits->push_back(vel_max);
    v_high_limits->push_back(accl_max);
    v_high_limits->push_back(accl_max);
}

void init_distributions(vector<Observation> *obs,
                        vector<Sample> *pi_0, vector<Sample> *m_0, vector<Sample> *v_0,
                        vector<Sample> *pi_1, vector<Sample> *m_1, vector<Sample> *v_1){

    //////////////////////////////////////////////////////////
    //////////////////////P_0/////////////////////////////////
    //////////////////////////////////////////////////////////
    if (!pi_0){
        LOG(FATAL) << "PI 0 vector was not initialized!";
    }

    for (size_t i = 0; i < 20; i++){
        Sample pi_temp;
        pi_temp.values.push_back(0.0);
        pi_temp.values.push_back(0.0);

        pi_0->push_back(pi_temp);
    }

    //////////////////////////////////////////////////////////
    //////////////////////P_1/////////////////////////////////
    //////////////////////////////////////////////////////////
    if (!pi_1){
        LOG(FATAL) << "PI 1 vector was not initialized!";
    }

    for (size_t i = 0; i < 20; i++){
        Sample pi_temp;
        pi_temp.values.push_back(0.0);
        pi_temp.values.push_back(0.0);

        pi_1->push_back(pi_temp);
    }

    //////////////////////////////////////////////////////////
    //////////////////////M_0/////////////////////////////////
    //////////////////////////////////////////////////////////
    if (!m_0){
        LOG(FATAL) << "M 0 vector was not initialized!";
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
        LOG(FATAL) << "Cannot open M 1 input file!";
    }

    file.close();

    //////////////////////////////////////////////////////////
    //////////////////////M_1/////////////////////////////////
    //////////////////////////////////////////////////////////
    if (!m_1){
        LOG(FATAL) << "M 1 vector was not initialized!";
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
        LOG(FATAL) << "Cannot open M 2 input file!";
    }

    file.close();

    //////////////////////////////////////////////////////////
    //////////////////////V_0/////////////////////////////////
    //////////////////////////////////////////////////////////
    if (!v_0){
        LOG(FATAL) << "V 0 vector was not initialized!";
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
        LOG(FATAL) << "Cannot open V 1 input file!";
    }

    file.close();

    //////////////////////////////////////////////////////////
    //////////////////////V_1/////////////////////////////////
    //////////////////////////////////////////////////////////
    if (!v_1){
        LOG(FATAL) << "V 1 vector was not initialized!";
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
        LOG(FATAL) << "Cannot open V 2 input file!";
    }

    file.close();

    //////////////////////////////////////////////////////////
    //////////////////////OBSERVATION/////////////////////////
    //////////////////////////////////////////////////////////
    if (!obs){
        LOG(FATAL) << "Observation vector was NULL!";
    }

    file.open("/home/sina/Desktop/obs.txt");

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

            sample_obs.values.pop_back();
            obs->push_back(sample_obs);
        }

    }else{
        LOG(FATAL) << "Cannot open Obs input file!";
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
