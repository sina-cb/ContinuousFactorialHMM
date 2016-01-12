#include "DETree.h"
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
using namespace std;

DETree::DETree(){
    LOG(INFO) << "The DETree is not created, you should use create_tree explicitly to initialize the DETree";
}

DETree::DETree(vector<Sample> &sample_set, vector<double> *sample_low, vector<double> *sample_high){
    create_tree(sample_set, sample_low, sample_high);
}

double DETree::density_value(Sample sample, double rho){

    DETreeNode *node = this->get_root();

    bool cond = !node->leaf_node;
    while(cond){
        int index = node->max_diff_index;

        cond = !node->leaf_node;

        if (cond && sample.values[index] < node->cut_value){
            node = node->left_child;
        }else if (cond){
            node = node->right_child;
        }
    }

    cond = !(node->node_type == 'R');
    double p = 0.0;
    double temp_rho = 1.0;
    while (cond){
        if (node->node_type != 'R'){
            p = p + (1 - rho) * temp_rho * node->node_sigma;
        } else {
            p = p + temp_rho * node->node_sigma;
        }
        temp_rho = temp_rho * rho;

        cond = !(node->node_type == 'R');
        node = node->parent;
    }

    return p;
}

void DETree::create_tree(const vector<Sample> &sample_set, vector<double> *sample_low, vector<double> *sample_high){
    samples_low_limit = sample_low;
    samples_high_limit = sample_high;
    root = new DETreeNode(sample_set, 0, 'R');
}

string DETree::depth_first_str(){
    stringbuf buf;
    ostream os (&buf);
    vector<DETreeNode*> * nodes = depth_first();
    for (size_t i = 0; i < nodes->size(); i++){
        os << "Level: " << (*nodes)[i]->level << " Side: ";
        os << (*nodes)[i]->node_type << " --> Size: ";
        os << (*nodes)[i]->node_size << " P: " << (*nodes)[i]->node_sigma << endl;
    }
    return buf.str();
}

vector<DETreeNode*>* DETree::depth_first(){
    vector<DETreeNode*>* results = new vector<DETreeNode*>();
    depth_first(results, root);
    return results;
}

void DETree::depth_first(vector<DETreeNode *> *& nodes, DETreeNode *current_node){
    if (current_node->leaf_node){
        nodes->push_back(current_node);
        return;
    }

    nodes->push_back(current_node);

    if (current_node->left_child != NULL){
        depth_first(nodes, current_node->left_child);
    }

    if (current_node->right_child != NULL){
        depth_first(nodes, current_node->right_child);
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

DETreeNode* DETree::get_root(){
    return root;
}

DETreeNode::DETreeNode(){

}

DETreeNode::DETreeNode(vector<Sample> sub_sample, int level, char node_type){

    // set the current node's sample set
    for (size_t i = 0; i < sub_sample.size(); i++){
        samples.push_back(sub_sample[i]);
        node_sigma = node_sigma + sub_sample[i].p;
    }

    node_size = samples.size();
    this->node_type = node_type;
    this->level = level;

    // Finding the subsets
    vector<double> min_vals;
    vector<double> max_vals;

    for (size_t i = 0; i < samples[0].size(); i++){
        min_vals.push_back(std::numeric_limits<double>::max());
        max_vals.push_back(std::numeric_limits<double>::min());
    }

    for (size_t i = 0; i < samples.size(); i++){
        for (size_t j = 0; j < samples[0].size(); j++){
            if (min_vals[j] > samples[i].values[j]){
                min_vals[j] = samples[i].values[j];
            }

            if (max_vals[j] < samples[i].values[j]){
                max_vals[j] = samples[i].values[j];
            }
        }
    }

    max_diff_index = 0;
    max_diff = abs(max_vals[0] - min_vals[0]);
    for (size_t i = 0; i < samples[0].size(); i++){
        if (abs(max_vals[i] - min_vals[i]) > abs(max_vals[max_diff_index] - min_vals[max_diff_index])){
            max_diff_index = i;
            max_diff = abs(max_vals[i] - min_vals[i]);
        }
    }

    max_diff_max_value = max_vals[max_diff_index];
    max_diff_min_value = min_vals[max_diff_index];

    if (samples.size() <= 1 || max_diff < min_diff_interval){
        leaf_node = true;
        return;
    }

    // divide the samples
    std::sort(samples.begin(), samples.end(),
              [&, this](const Sample& a, const Sample& b) {
                    return a.values[max_diff_index] < b.values[max_diff_index];
                }
    );

    cut_index = 0;
    cut_value = samples[0].values[max_diff_index] + max_diff / 2.0;
    for (size_t i = 0; i < samples.size(); i++){
        if (samples[i].values[max_diff_index] < cut_value){
            cut_index = i;
        }
    }

    if (cut_index == (int)samples.size() - 1){
        cut_index--;
    }

    vector<Sample> sub_sample_left (
                samples.begin(),
                samples.begin() + cut_index + 1
                );
    vector<Sample> sub_sample_right(
                samples.begin() + cut_index + 1,
                samples.end()
                );

    // calculate the probability for the current node


    this->left_child = new DETreeNode(sub_sample_left, level + 1, 'l');
    this->right_child = new DETreeNode(sub_sample_right, level + 1, 'r');

    this->left_child->parent = this;
    this->right_child->parent = this;

}

string DETreeNode::str(){
    stringbuf buf;
    ostream os(&buf);
    for (size_t i = 0; i < samples.size(); i++){
        for (size_t j = 0; j < samples[0].size(); j++){
            os << samples[i].values[j] << "\t";
        }
        os << "P: " << samples[i].p << endl;
    }

    return buf.str();
}
