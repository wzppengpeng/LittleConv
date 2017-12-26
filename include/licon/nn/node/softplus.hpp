#ifndef LICON_NN_NODE_SOFTPLUS_HPP_
#define LICON_NN_NODE_SOFTPLUS_HPP_

/**
 * the softplus function, to make value into plus
 */
#include "licon/nn/node/neuron.hpp"


namespace licon
{

namespace nn
{

class Softplus : public NeuronNode<F> {

public:
    // the heap create funtion
    static nn::NodePtr CreateSoftplus(F beta = 1.0, F threshold = 20.0);

    // the name
    virtual inline std::string name() const { return "softplus"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data;

    // the parameters of this activation function
    F m_beta;
    F m_threshold;

private:
    // constructor
    Softplus(F beta, F threshold);

};

} //nn

} //licon


#endif /*LICON_NN_NODE_SOFTPLUS_HPP_*/