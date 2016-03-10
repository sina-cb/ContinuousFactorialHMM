#include "Observation.h"

size_t Observation::size(){
    return values.size();
}

Observation Observation::convert(Sample sample){
    Observation result;
    for (size_t i = 0; i < sample.values.size(); i++){
        result.values.push_back(sample.values[i]);
    }
    return result;
}

Sample Observation::combine(Sample second){
    Sample result;

    for (size_t i = 0; i < this->size(); i++){
        result.values.push_back(values[i]);
    }
    for (size_t i = 0; i < second.size(); i++){
        result.values.push_back(second.values[i]);
    }

    return result;
}
