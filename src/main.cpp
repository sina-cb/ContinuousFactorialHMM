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

vector<string> hmm_types = {"Monte Carlo HMM", "Layered Monte Carlo HMM"};

void init_GLOG();

int main(int argc, char** argv){
    init_GLOG();

    LMCHMM * hmm = new LMCHMM(2);
    LOG(INFO) << "Number of levels: " << hmm->_layers_count();

    hmm->learn_hmm(NULL, 0, 0);
    hmm->forward(new vector<Observation>(), 100);
    hmm->most_probable_seq();

    return 0;
}

void init_GLOG(){
    InitGoogleLogging(((string)hmm_types[HMM_TYPE]).c_str());
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");
}
