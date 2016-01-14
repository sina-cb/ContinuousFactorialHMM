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
using namespace std;
using namespace google;

void init_pi(vector<Sample> * pi, int sample_count);
void init_m(vector<Sample> * m, int sample_count);
void init_v(vector<Sample> * v, int sample_count);
void init_limits(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                 vector<double> * m_low_limits, vector<double> * m_high_limits,
                 vector<double> * v_low_limits, vector<double> * v_high_limits);
void init_observations(vector<Observation> * obs, size_t size);

void init_GLOG(int argc, char* argv[]);
void print(vector<Sample> dist);

void run_small_tree_creation();
vector<Sample> fixed_sample_set();

int main(int argc, char* argv[])
{
    init_GLOG(argc, argv);

    // Gathering samples from the distributions
    vector<Sample> *pi = new vector<Sample>();
    vector<Sample> *m = new vector<Sample>();
    vector<Sample> *v = new vector<Sample>();
    {
        Timer tmr;
        double t1 = tmr.elapsed();

        init_pi(pi, 20);
        init_m(m, 1000);
        init_v(v, 1000);

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
        Timer tmr;
        double t1 = tmr.elapsed();

        init_observations(obs, 10000);
        vector<vector<Sample> > forward = hmm.forward(obs, 100);

        int tr = 0;
        for (size_t i = 0; i < forward.size(); i++){
            double state0 = 0.0;
            double state1 = 0.0;
            for (size_t j = 0; j < forward[i].size(); j++){
                //LOG(INFO) << "Forward: " << i << " " << j << " " << forward[i][j].values[0];
                if (forward[i][j].values[0] <= 1.5){
                    state0 += forward[i][j].p;
                }else{
                    state1 += forward[i][j].p;
                }
            }

            cout << "Observation " << i << ":\t" << (*obs)[i].values[0] << endl;
            cout << "State 0: " << state0 << " State 1: " << state1 << endl << endl;

            if (i % 2 == 1){
                if (state0 < state1){
                    tr++;
                }
            }else{
                if (state1 < state0){
                    tr++;
                }
            }
        }

        LOG(INFO) << "True: " << tr << endl;

        double t2 = tmr.elapsed();
        LOG(INFO) << "Testing the MCFHMM time: " << (t2 - t1) << " seconds";
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Initialization Part ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void init_pi(vector<Sample> *pi, int sample_count){
    if (!pi){
        LOG(FATAL) << "PI vector was not initialized!";
    }

    for (int i = 0; i < sample_count; i++){
        Sample sample;
        sample.values.push_back(1);
        sample.p = 1.0 / sample_count;
        pi->push_back(sample);
    }
}

void init_m(vector<Sample> *m, int sample_count){
    if (!m){
        LOG(FATAL) << "M vector was not initialized!";
    }

    for (int i = 0; i < sample_count / 2; i++){
        Sample sample1;
        sample1.values.push_back(1);
        sample1.values.push_back(2);
        sample1.p = 1.0 / sample_count;
        m->push_back(sample1);

        Sample sample2;
        sample2.values.push_back(2);
        sample2.values.push_back(1);
        sample2.p = 1.0 / sample_count;
        m->push_back(sample2);
    }
}

void init_v(vector<Sample> *v, int sample_count){
    if (!v){
        LOG(FATAL) << "V vector was not initialized!";
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(-.1, .1);

    double x = 5, y = 5;

    for (int i = 0; i < sample_count / 2; i++){
        Sample sample1;
        sample1.values.push_back(.1 + dist(gen));
        sample1.values.push_back(.1 + dist(gen));
        sample1.values.push_back(1);
        sample1.p = 1.0 / sample_count;
        v->push_back(sample1);

        Sample sample2;
        sample2.values.push_back(x + dist(gen));
        sample2.values.push_back(y + dist(gen));
        sample2.values.push_back(2);
        sample2.p = 1.0 / sample_count;
        v->push_back(sample2);
    }
}

void init_limits(vector<double> * pi_low_limits, vector<double> * pi_high_limits,
                 vector<double> * m_low_limits, vector<double> * m_high_limits,
                 vector<double> * v_low_limits, vector<double> * v_high_limits){

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

void init_observations(vector<Observation> * obs, size_t size){
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
