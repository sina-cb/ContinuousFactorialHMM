#include <iostream>
#include "Sampler.h"
#include "Sample.h"
#include "MCFHMM.h"
#include "DETree.h"
#include <glog/logging.h>
using namespace std;
using namespace google;

int main(int argc, char* argv[])
{
    InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 0;
    FLAGS_log_dir = ".";
    FLAGS_minloglevel = 0;
    //FLAGS_logtostderr = true;
    //SetLogDestination(google::INFO, "./info");

    Sampler sampler;
    MCFHMM hmm;
    DETree pi_tree;

    hmm.init_hmm(30, 100, 100);

    vector<Sample> *pi = hmm.get_pi();

    sampler.likelihood_weighted_sampler(*pi);

    pi_tree.create_tree(*pi);
    LOG(INFO) << "Depth First Tree: " << endl << pi_tree.depth_first_str();

    pi = sampler.resample_from(&pi_tree, pi->size());

//    root.print_samples();
//    pi_tree.left_child(root).print_samples();

    return 0;
}

