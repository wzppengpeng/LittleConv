#ifndef LICON_NN_NEURON_NODE_HPP_
#define LICON_NN_NEURON_NODE_HPP_

#include "licon/nn/operation_node.hpp"

/**
 * the basic single pass neuron node(which only one-pass-one)
 */


namespace licon
{

namespace nn
{

template<typename Dtype>
class NeuronNode : public OpNode<Dtype> {
public:
    // construct
    NeuronNode() : OpNode<Dtype>() {}

    // deconstructor
    virtual ~NeuronNode() {}

    // the nums of pre node
    virtual inline int exact_num_former_nodes() const { return 1; }

    // the nums of post node
    virtual inline int exact_num_after_nodes() const  { return 1; }

protected:
    virtual inline void Resize() {
        OpNode<Dtype>::m_former_nodes.resize(1, nullptr);
        OpNode<Dtype>::m_after_nodes.resize(1, nullptr);
    }

};

} //nn

} //licon


#endif /*LICON_NN_NEURON_NODE_HPP_*/