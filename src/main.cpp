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

    double min_s = 0;
    double max_s = 1;

    double min_o = 6;
    double max_o = 14;

    pi_low_limits->push_back(min_s);

    pi_high_limits->push_back(max_s);

    m_low_limits->push_back(min_s);
    m_low_limits->push_back(min_s);

    m_high_limits->push_back(max_s);
    m_high_limits->push_back(max_s);

    v_low_limits->push_back(min_o);
    v_low_limits->push_back(min_s);

    v_high_limits->push_back(max_o);
    v_high_limits->push_back(max_s);
}

void init_limits_hmm_1(vector<double> *pi_low_limits, vector<double> *pi_high_limits,
                       vector<double> *m_low_limits, vector<double> *m_high_limits,
                       vector<double> *v_low_limits, vector<double> *v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    double min_s = 0;
    double max_s = 1;

    double min_o = 0;
    double max_o = 1;

    pi_low_limits->push_back(min_s);

    pi_high_limits->push_back(max_s);

    m_low_limits->push_back(min_s);
    m_low_limits->push_back(min_s);

    m_high_limits->push_back(max_s);
    m_high_limits->push_back(max_s);

    v_low_limits->push_back(min_o);
    v_low_limits->push_back(min_s);

    v_high_limits->push_back(max_o);
    v_high_limits->push_back(max_s);
}

void init_distributions(vector<Observation> *obs,
                        vector<Sample> *pi_0, vector<Sample> *m_0, vector<Sample> *v_0,
                        vector<Sample> *pi_1, vector<Sample> *m_1, vector<Sample> *v_1){

    double min_s_0 = 0;
    double max_s_0 = 1;
    double min_s_1 = 0;
    double max_s_1 = 1;

    // Initializing the PI distributions
    for (size_t i = 0; i < 100; i++){
        Sample _0;
        _0.values.push_back(min_s_0);
        _0.p = 1.0 / 100;
        pi_0->push_back(_0);

        Sample _1;
        _1.values.push_back(min_s_1);
        _1.p = 1.0 / 100;
        pi_1->push_back(_1);
    }

    // Generate observations and the states generating them
    double current_s_0 = 0;
    double current_s_1 = 0;
    for (size_t i = 0; i < OBSERVATION_COUNT; i++){

        double old_state_0 = current_s_0;
        double old_state_1 = current_s_1;

        if (current_s_1 == min_s_1){
            // determine the new state of the upper number generator
            double tice = drand48() * 100;
            if (tice < 20){
                current_s_1 = max_s_1;
            }else{
                current_s_1 = current_s_1;
            }
        }else{ // if (current_s_1 == max_s_1){
            // determine the new state of the upper number generator
            double tice = drand48() * 100;
            if (tice < 20){
                current_s_1 = min_s_1;
            }else{
                current_s_1 = current_s_1;
            }
        }

        // Now based on the new state, determine the lower level state
        double tice = drand48() * 100;
        if (current_s_1 == min_s_1 && tice < 0.2){
            current_s_0 = min_s_0;
        }else if (current_s_1 == min_s_1 && tice>= 0.2){
            current_s_0 = max_s_0;
        }else if (current_s_1 == max_s_1 && tice < 0.2){
            current_s_0 = max_s_0;
        }else if (current_s_1 == max_s_1 && tice >= 0.2){
            current_s_0 = min_s_0;
        }

        // Now current state_0 and state_1 are initialized
        // Let's make some observations
        double num_1 = 0;
        double num_2 = 0;
        if (current_s_1 == min_s_1 && current_s_0 == min_s_0){
            double rand = (drand48() * 0.2) - 0.1;
            num_1 = 1 + rand;

            rand = (drand48() * 0.2) - 0.1;
            num_2 = 6 + rand;
        }else if (current_s_1 == min_s_1 && current_s_0 == max_s_0){
            double rand = (drand48() * 0.2) - 0.1;
            num_1 = 3 + rand;

            rand = (drand48() * 0.2) - 0.1;
            num_2 = 6 + rand;
        }else if (current_s_1 == max_s_1 && current_s_0 == min_s_0){
            double rand = (drand48() * 0.2) - 0.1;
            num_1 = 1 + rand;

            rand = (drand48() * 0.2) - 0.1;
            num_2 = 10 + rand;
        }else if (current_s_1 == max_s_1 && current_s_0 == max_s_0){
            double rand = (drand48() * 0.2) - 0.1;
            num_1 = 3 + rand;

            rand = (drand48() * 0.2) - 0.1;
            num_2 = 10 + rand;
        }

        Observation obs_s;
        obs_s.values.push_back(num_1 + num_2);

        Sample m_1_s;
        m_1_s.values.push_back(old_state_0);
        m_1_s.values.push_back(current_s_0);
        m_1_s.p = 1.0 / OBSERVATION_COUNT;

        Sample m_2_s;
        m_2_s.values.push_back(old_state_1);
        m_2_s.values.push_back(current_s_1);
        m_2_s.p = 1.0 / OBSERVATION_COUNT;

        Sample v_1_s;
        v_1_s.values.push_back(num_1 + num_2);
        v_1_s.values.push_back(current_s_0);
        v_1_s.p = 1.0 / OBSERVATION_COUNT;

        Sample v_2_s;
        v_2_s.values.push_back(current_s_0);
        v_2_s.values.push_back(current_s_1);
        v_2_s.p = 1.0 / OBSERVATION_COUNT;

        // Add the samples to the distribution
        obs->push_back(obs_s);

        m_0->push_back(m_1_s);
        m_1->push_back(m_2_s);

        v_0->push_back(v_1_s);
        v_1->push_back(v_2_s);
    }

    //    LOG(INFO) << "Sizes:";
    //    LOG(INFO) << "\n"
    //              << "\tObs:  " << obs->size() << "\n"
    //              << "\tPI_0: " << pi_0->size() << "\n"
    //              << "\tM_0:  " << m_0->size() << "\n"
    //              << "\tV_0:  " << v_0->size() << "\n"
    //              << "\tPI_1: " << pi_1->size() << "\n"
    //              << "\tM_1:  " << m_1->size() << "\n"
    //              << "\tV_1:  " << v_1->size() << "\n";

}

void init_GLOG(){
    InitGoogleLogging(((string)hmm_types[HMM_TYPE]).c_str());
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 1;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}
