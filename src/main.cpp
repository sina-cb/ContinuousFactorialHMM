#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <glog/logging.h>
#include "Sampler.h"
#include "Sample.h"
#include "MCFHMM.h"
#include "DETree.h"
#include "Observation.h"
using namespace std;
using namespace google;

void init_GLOG(int argc, char* argv[]);
void init_limits();
void init_observations();

vector<double> pi_low_limits;
vector<double> pi_high_limits;
vector<double> m_low_limits;
vector<double> m_high_limits;
vector<double> v_low_limits;
vector<double> v_high_limits;
vector<Observation> obs;

int main(int argc, char* argv[])
{
    init_GLOG(argc, argv);

    init_limits();
    init_observations();

    MCFHMM hmm;

    int N = 5;
    int max_iteration = 2;

    hmm.set_limits(&pi_low_limits, &pi_high_limits, &m_low_limits, &m_high_limits, &v_low_limits, &v_high_limits);
    hmm.learn_hmm(&obs, max_iteration, N);

//    root.print_samples();
//    pi_tree.left_child(root).print_samples();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// Initialization Part ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

void init_GLOG(int argc, char* argv[]){
    InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 1;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}

void init_observations(){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    uniform_real_distribution<double> dist(-.1, 0.1);
    double x = 0;
    double y = 0;
    double th = 0;
    for (size_t i = 0; i < 100; i++){
        Observation temp1;
        temp1.values.push_back(x + dist(gen));
        temp1.values.push_back(y + dist(gen));
        temp1.values.push_back(th + dist(gen));
        obs.push_back(temp1);

        Observation temp2;
        temp2.values.push_back( x + 10.0 + dist(gen));
        temp2.values.push_back( y + 10.0 + dist(gen));
        temp2.values.push_back(th + 10.0 + dist(gen));
        obs.push_back(temp2);
    }
}

void init_limits(){
    pi_low_limits.push_back(0);
    pi_low_limits.push_back(0);
    pi_low_limits.push_back(0);

    pi_high_limits.push_back(10);
    pi_high_limits.push_back(10);
    pi_high_limits.push_back(2 * M_PI);

    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);
    m_low_limits.push_back(0);

    m_high_limits.push_back(10);
    m_high_limits.push_back(10);
    m_high_limits.push_back(2 * M_PI);
    m_high_limits.push_back(10);
    m_high_limits.push_back(10);
    m_high_limits.push_back(2 * M_PI);

    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);
    v_low_limits.push_back(0);

    v_high_limits.push_back(10);
    v_high_limits.push_back(10);
    v_high_limits.push_back(2 * M_PI);
    v_high_limits.push_back(10);
    v_high_limits.push_back(10);
    v_high_limits.push_back(2 * M_PI);
}















