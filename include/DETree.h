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
    DETree(vector<Sample> const &sample_set);

    void create_tree(const vector<Sample> &sample_set);

    DETreeNode get_root();

    vector<DETreeNode*>* depth_first();
    string depth_first_str();

private:
    DETreeNode *root;
    void depth_first(vector<DETreeNode*> *& nodes, DETreeNode* current_node);

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

    double cut_value;
    int cut_index;
    int max_diff_index;
    double max_diff;

    DETreeNode *left_child;
    DETreeNode *right_child;
};

#endif
