#include <vector>

using namespace std;

#ifndef DETREE_H
#define DETREE_H

class DETreeNode;

class DETree{

public:
    DETree();
    DETree(vector< vector<double> > const &sample_set);

    void create_tree(const vector< vector<double> > &sample_set);

    DETreeNode get_root();
    DETreeNode left_child  (DETreeNode parent);
    DETreeNode right_child (DETreeNode parent);

private:
    DETreeNode *root;

};

class DETreeNode{

public:
    DETreeNode();
    DETreeNode(vector< vector<double> >sub_sample);

    void print_samples();

    bool leaf_node = false;

    DETreeNode *left_child;
    DETreeNode *right_child;

    vector< vector <double> > samples;
    double probability = 0;

};

#endif
