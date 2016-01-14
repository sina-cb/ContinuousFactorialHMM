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

int main(int argc, char* argv[])
{
    init_GLOG(argc, argv);

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

    vector<double> * low_limits = new vector<double>();
    vector<double> * high_limits = new vector<double>();
    low_limits->push_back(0);
    low_limits->push_back(0);
    high_limits->push_back(10);
    high_limits->push_back(10);

    DETree tree(samples, low_limits, high_limits);

    LOG(INFO) << "\n" << tree.depth_first_str();

    return 0;

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
