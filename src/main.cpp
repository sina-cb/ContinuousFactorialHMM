#include <iostream>
#include "Sampler.h"
#include "MCFHMM.h"
#include "DETree.h"

using namespace std;

int main()
{
    Sampler sampler;
    MCFHMM hmm;
    DETree pi_tree;

    hmm.init_hmm(10, 100, 100);

    vector<pi_type> *pi = hmm.get_pi();

    //    for (unsigned int i = 0; i < pi->size(); i++){
    //        cout << ((*pi)[i])[0] << endl;
    //    }

    cout << endl;

    sampler.likelihood_weighted_sampler(*pi);

    pi_tree.create_tree(*pi);

    DETreeNode root = pi_tree.get_root();

    root.print_samples();
    pi_tree.left_child(root).print_samples();

    return 0;
}

