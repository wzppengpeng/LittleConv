#ifndef LICON_NN_POOL_HPP_
#define LICON_NN_POOL_HPP_

#include "licon/nn/node/neuron.hpp"

/**
 * the basic interface for pooling layers
 * will need a kernel size(may be window size or final output size)
 */


namespace licon
{

namespace nn
{

template<typename Dtype>
class PoolNode : public NeuronNode<Dtype> {
public:
    // construct
    PoolNode(const int kernel_size) : NeuronNode<Dtype>(), m_kernel_size(kernel_size) {}

    // deconstruct
    virtual ~PoolNode() {}

protected:
    // the kernel size
    int m_kernel_size;

};

} //nn

} //licon

#endif /*LICON_NN_POOL_HPP_*/