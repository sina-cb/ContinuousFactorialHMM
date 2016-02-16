#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <glog/logging.h>
#include "Sampler.h"
#include "Sample.h"
#include "MCFHMM.h"
#include "DETree.h"
#include "Observation.h"
#include "Timer.h"
#include <fstream>
#include <sstream>
using namespace std;
using namespace google;

#define N 100
#define N_HMM_TEST 100
#define MAX_ITERATION 20
#define INIT_OBS_C 100
#define TEST_OBS_C 50
#define PI_SAMPLE_C 20
#define M_SAMPLE_C 200
#define V_SAMPLE_C 200

#define USE_EM 1

void init_pi(vector<Sample> * pi, int sample_count);
void init_m(vector<Sample> * m, int sample_count);
void init_v(vector<Sample> * v, int sample_count);
void init_limits(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                 vector<double> * m_low_limits, vector<double> * m_high_limits,
                 vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_observations(vector<Observation> * obs, size_t size);

void init_GLOG(int argc, char* argv[]);
vector<Sample> fixed_sample_set();
void print(vector<Sample> dist);

void use_precollected_samples();
void use_em_learning();
void run_small_tree_creation();

int main(int argc, char* argv[])
{
    init_GLOG(argc, argv);

#if USE_EM
    use_em_learning();
#else
    use_precollected_samples();
#endif

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Initialization Part ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void use_em_learning(){

    // Gathering samples from the distributions
    vector<Sample> *pi = new vector<Sample>();
    vector<Sample> *m = new vector<Sample>();
    vector<Sample> *v = new vector<Sample>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_pi(pi, PI_SAMPLE_C);
        init_m(m, M_SAMPLE_C);
        init_v(v, V_SAMPLE_C);

        LOG(INFO) << "PI size: " << pi->size();
        LOG(INFO) << "M size: " << m->size();
        LOG(INFO) << "V size: " << v->size();

        double t2 = tmr.elapsed();

        LOG(INFO) << "Generating time: " << (t2 - t1) << " seconds";
    }

    // Initializing the limits for the values that each variable can take
    vector<double> * pi_low_limits = new vector<double>();
    vector<double> * pi_high_limits = new vector<double>();
    vector<double> * m_low_limits = new vector<double>();
    vector<double> * m_high_limits = new vector<double>();
    vector<double> * v_low_limits = new vector<double>();
    vector<double> * v_high_limits = new vector<double>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_limits(pi_low_limits, pi_high_limits, m_low_limits, m_high_limits, v_low_limits, v_high_limits);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the limits time: " << (t2 - t1) << " seconds";
    }

    vector<Observation> *observations = new vector<Observation>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_observations(observations, INIT_OBS_C);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the observations time: " << (t2 - t1) << " seconds";
    }

    // Generating the HMM from the gathered samples and the known limits
    MCFHMM hmm;
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        hmm.set_limits(pi_low_limits, pi_high_limits, m_low_limits, m_high_limits, v_low_limits, v_high_limits);
        hmm.set_distributions(pi, m, v, 0.5);
        hmm.learn_hmm(observations, 4, N);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Generating the MCFHMM time: " << (t2 - t1) << " seconds";
    }

    //Testing the accuracy
    vector<Observation> * obs = new vector<Observation>;
    {
        Sampler sampler;

        Timer tmr;
        double t1 = tmr.elapsed();

        //         int tr = 0;
//                for (size_t i = 1; i < TEST_OBS_C; i++){
//        obs->push_back((*observations)[observations->size() - 3]);
//        obs->push_back((*observations)[observations->size() - 2]);
        obs->push_back((*observations)[7]);
        DETree forward = hmm.forward(obs, N_HMM_TEST);

        Sample big_sample = sampler.sample(&forward);
        for (size_t i = 0; i < 5; i++){
            Sample temp = sampler.sample(&forward);
            LOG(INFO) << "Temp " << i << ":\t"
                      << temp.values[0] << "\t"
                      << temp.values[1];

            big_sample.values[0] = big_sample.values[0] + temp.values[0];
            big_sample.values[1] = big_sample.values[1] + temp.values[1];
        }

        big_sample.values[0] /= 6;
        big_sample.values[1] /= 6;

        LOG(INFO) << "Big Sample:\t"
                  << big_sample.values[0] << "\t"
                  << big_sample.values[1];
        //        }

        //         LOG(INFO) << "Accuracy: " << ((tr / (double) TEST_OBS_C) * 100.0) << "%" << endl;

        double t2 = tmr.elapsed();
        LOG(INFO) << "Testing the MCFHMM time: " << (t2 - t1) << " seconds";
    }

}

void use_precollected_samples(){
    // Gathering samples from the distributions
    vector<Sample> *pi = new vector<Sample>();
    vector<Sample> *m = new vector<Sample>();
    vector<Sample> *v = new vector<Sample>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_pi(pi, PI_SAMPLE_C);
        init_m(m, M_SAMPLE_C);
        init_v(v, V_SAMPLE_C);

        LOG(INFO) << "PI size: " << pi->size();
        LOG(INFO) << "M size: " << m->size();
        LOG(INFO) << "V size: " << v->size();

        double t2 = tmr.elapsed();

        LOG(INFO) << "Generating time: " << (t2 - t1) << " seconds";
    }

    // Initializing the limits for the values that each variable can take
    vector<double> * pi_low_limits = new vector<double>();
    vector<double> * pi_high_limits = new vector<double>();
    vector<double> * m_low_limits = new vector<double>();
    vector<double> * m_high_limits = new vector<double>();
    vector<double> * v_low_limits = new vector<double>();
    vector<double> * v_high_limits = new vector<double>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_limits(pi_low_limits, pi_high_limits, m_low_limits, m_high_limits, v_low_limits, v_high_limits);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Initializing the limits time: " << (t2 - t1) << " seconds";
    }

    // Generating the HMM from the gathered samples and the known limits
    MCFHMM hmm;
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        hmm.set_limits(pi_low_limits, pi_high_limits, m_low_limits, m_high_limits, v_low_limits, v_high_limits);
        hmm.set_distributions(pi, m, v, 0.5);

        double t2 = tmr.elapsed();
        LOG(INFO) << "Generating the MCFHMM time: " << (t2 - t1) << " seconds";
    }

    // Testing the accuracy
    vector<Observation> * obs = new vector<Observation>;
    {
        Sampler sampler;

        Timer tmr;
        double t1 = tmr.elapsed();

        //         int tr = 0;
        //        for (size_t i = 1; i < TEST_OBS_C; i++){
        init_observations(obs, -1);
        DETree forward = hmm.forward(obs, N_HMM_TEST);

        Sample sample = sampler.sample(&forward);
        LOG(INFO) << "Sample:\t"
                  << sample.values[0] << "\t"
                  << sample.values[1] << "\t"
                  << sample.values[2];
        //        }

        //         LOG(INFO) << "Accuracy: " << ((tr / (double) TEST_OBS_C) * 100.0) << "%" << endl;

        double t2 = tmr.elapsed();
        LOG(INFO) << "Testing the MCFHMM time: " << (t2 - t1) << " seconds";
    }
}

void init_pi(vector<Sample> *pi, int sample_count){
    if (!pi){
        LOG(FATAL) << "PI vector was not initialized!";
    }

    for (size_t i = 0; i < 20; i++){
        Sample pi_temp;
        pi_temp.values.push_back(0.0);
        pi_temp.values.push_back(0.0);

        pi->push_back(pi_temp);
    }
}

void init_m(vector<Sample> *m, int sample_count){
    if (!m){
        LOG(FATAL) << "M vector was not initialized!";
    }

    ifstream file("/home/sina/Desktop/m.txt");

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

            m->push_back(sample_m);
        }

    }else{
        LOG(FATAL) << "Cannot open M input file!";
    }

    file.close();

}

void init_v(vector<Sample> *v, int sample_count){
    if (!v){
        LOG(FATAL) << "V vector was not initialized!";
    }

    ifstream file("/home/sina/Desktop/v.txt");

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

            v->push_back(sample_v);
        }

    }else{
        LOG(FATAL) << "Cannot open V input file!";
    }

    file.close();
}

void init_limits(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                 vector<double> * m_low_limits, vector<double> * m_high_limits,
                 vector<double> * v_low_limits, vector<double> * v_high_limits){

    if (!pi_low_limits || !pi_high_limits
            || !m_low_limits || !m_high_limits
            || !v_low_limits || !v_high_limits){
        LOG(FATAL) << "One of the limit vectors is NULL!";
    }

    // TODO: Add bounds to the vectors here!
    double accel_min = -0.4;

    double accel_max = 0.4;

    double crosswalk_min = 0;
    double crosswalk_max = 1;

    double turn_point_min = 0;
    double turn_point_max = 2;

    double junction_min = 0;
    double junction_max = 2;

    double wall_min = 0;
    double wall_max = 3.0;

    ////////// INIT PI BOUNDS //////////
    pi_low_limits->push_back(accel_min);
    pi_low_limits->push_back(accel_min);

    pi_high_limits->push_back(accel_max);
    pi_high_limits->push_back(accel_max);


    ////////// INIT M  BOUNDS //////////
    m_low_limits->push_back(accel_min);
    m_low_limits->push_back(accel_min);
    m_low_limits->push_back(accel_min);
    m_low_limits->push_back(accel_min);

    m_high_limits->push_back(accel_max);
    m_high_limits->push_back(accel_max);
    m_high_limits->push_back(accel_max);
    m_high_limits->push_back(accel_max);


    ////////// INIT V  BOUNDS //////////
    v_low_limits->push_back(crosswalk_min);
    v_low_limits->push_back(turn_point_min);
    v_low_limits->push_back(junction_min);
    v_low_limits->push_back(wall_min);
    v_low_limits->push_back(wall_min);
    v_low_limits->push_back(accel_min);
    v_low_limits->push_back(accel_min);

    v_high_limits->push_back(crosswalk_max);
    v_high_limits->push_back(turn_point_max);
    v_high_limits->push_back(junction_max);
    v_high_limits->push_back(wall_max);
    v_high_limits->push_back(wall_max);
    v_high_limits->push_back(accel_max);
    v_high_limits->push_back(accel_max);
}

void init_observations(vector<Observation> * obs, size_t size){
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

void print(vector<Sample> dist){
    for (size_t i = 0; i < dist.size(); i++){
        std::stringbuf buffer;
        std::ostream os (&buffer);
        for (size_t j = 0; j < dist[i].values.size(); j++){
            os << std::setprecision(3) << dist[i].values[j] << "\t";
        }
        LOG(INFO) << buffer.str();
    }
}

void init_GLOG(int argc, char* argv[]){
    InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}

void run_small_tree_creation(){
    vector<Sample> samples = fixed_sample_set();

    vector<double> * low_limits = new vector<double>();
    vector<double> * high_limits = new vector<double>();
    low_limits->push_back(0);
    low_limits->push_back(0);
    high_limits->push_back(10);
    high_limits->push_back(10);

    DETree tree(samples, low_limits, high_limits);

    Sample *s = new Sample();
    s->values.push_back(8.5);
    s->values.push_back(1);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    Sampler sampler;

    int probability = 0;

    for (int i = 0; i < 100000; i++){
        Sample y_sample;
        y_sample.values.push_back(1);
        Sample sample = sampler.sample_given(&tree, y_sample);
        //LOG(INFO) << sample.str();
        //LOG(INFO) << tree.density_value(sample, 0.8);
        if (sample.values[0] >= 2.5 && sample.values[0] < 5.0) probability++;
    }
    LOG(INFO) << probability;

    //    for (int i = 0; i < 100000; i++){
    //        Sample sample = sampler.sample(&tree);
    //        //LOG(INFO) << sample.str();
    //        //LOG(INFO) << tree.density_value(sample, 0.8);
    //        if (sample.values[0] >= 2.5 && sample.values[0] < 5.0) probability++;
    //    }

    //    LOG(INFO) << probability;

    //    LOG(INFO) << "\n" << tree.depth_first_str();
}

vector<Sample> fixed_sample_set(){
    vector<Sample> samples;

    Sample *s = new Sample();
    s->values.push_back(1.25);
    s->values.push_back(7);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(3);
    s->values.push_back(8);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(2);
    s->values.push_back(4);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(3.5);
    s->values.push_back(1.5);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(4);
    s->values.push_back(4);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(7);
    s->values.push_back(8);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(9);
    s->values.push_back(6);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    s = new Sample();
    s->values.push_back(8.5);
    s->values.push_back(1);
    s->p = 1.0 / 8.0;
    samples.push_back(*s);

    return samples;
}
