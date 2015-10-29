#include "DETree.h"
#include <iostream>
using namespace std;

DETree::DETree(){

}

DETree::DETree(const vector<vector<double> > &sample_set){
    create_tree(sample_set);
}

void DETree::create_tree(const vector<vector<double> > &sample_set){

    root = new DETreeNode(sample_set);

}

DETreeNode DETree::get_root(){
    return *(root);
}

DETreeNode DETree::left_child (DETreeNode parent){
    return *(parent.left_child);
}

DETreeNode DETree::right_child(DETreeNode parent){
    return *(parent.right_child);
}

DETreeNode::DETreeNode(){

}

DETreeNode::DETreeNode(vector<vector<double> > sub_sample){

    if (sub_sample.size() == 0){
        leaf_node = true;
        return;
    }

    // set the current nodes sample set
    for (int i = 0; i < sub_sample.size(); i++){
        samples.push_back(sub_sample[i]);
    }

    // divide the samples
    vector<vector<double> > sub_sample_left;
    vector<vector<double> > sub_sample_right;

    // calculate the probability for the current node


    this->left_child = new DETreeNode(sub_sample_left);
    this->right_child = new DETreeNode(sub_sample_right);

}

void DETreeNode::print_samples(){
    for (size_t i = 0; i < samples.size(); i++){
        for (size_t j = 0; j < samples[0].size(); j++){
            cout << samples[i][j] << "\t";
        }
        cout << endl;
    }
}
