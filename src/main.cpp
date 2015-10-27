#include <iostream>
#include "Sampler.h"
#include "MCFHMM.h"
#include "DETree.h"

using namespace std;

int main()
{

    MCFHMM hmm;

    hmm.init_hmm(30, 100, 100);

    vector<pi_type> *pi = hmm.get_pi();

    for (int i = 0; i < pi->size(); i++){
        cout << get<0>((*pi)[i]) << ", " << get<1>((*pi)[i]) << ", " << get<2>((*pi)[i]) << endl;
    }

}

