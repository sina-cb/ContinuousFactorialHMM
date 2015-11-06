#include <vector>
#include <glog/logging.h>
#include "Sample.h"
using namespace std;

#ifndef DETREE_H
#define DETREE_H

class DETreeNode;

class DETree{

public:
    DETree();
    DETree(vector<Sample> const &sample_set, vector<double> *sample_low, vector<double> *sample_high);

    void create_tree(const vector<Sample> &sample_set, vector<double> *sample_low, vector<double> *sample_high);
    double density_value(Sample sample);

    DETreeNode* get_root();

    vector<DETreeNode*>* depth_first();
    string depth_first_str();

    vector<double> *samples_low_limit;
    vector<double> *samples_high_limit;


private:
    DETreeNode *root;

    void depth_first(vector<DETreeNode*> *& nodes, DETreeNode* current_node);
    double rho = 0.2;
};

class DETreeNode{

public:
    DETreeNode();
    DETreeNode(vector<Sample>sub_sample, int level, char node_type);

    string str();

    vector<Sample> samples;

    bool leaf_node = false;
    int level = 0;
    int node_size = 0;
    char node_type = 'R';
    double node_sigma = 0.0;

    double max_diff_max_value;
    double max_diff_min_value;

    double cut_value;
    int cut_index;
    int max_diff_index;
    double max_diff;

    DETreeNode *left_child;
    DETreeNode *right_child;
    DETreeNode *parent;
};

#endif
