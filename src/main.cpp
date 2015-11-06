#include <iostream>
#include <cmath>
#include "Sampler.h"
#include "Sample.h"
#include "MCFHMM.h"
#include "DETree.h"
#include "Observation.h"
#include <glog/logging.h>
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

    int N = 100;
    int max_iteration = 2;

    hmm.set_limits(&pi_low_limits, &pi_high_limits, &m_low_limits, &m_high_limits, &v_low_limits, &v_high_limits);
    hmm.init_hmm(N, N, N);
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
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}

void init_observations(){

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
