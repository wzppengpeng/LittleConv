#ifndef LICON_RELU_NODE_HPP_
#define LICON_RELU_NODE_HPP_

/**
 * the relu active node
 */
#include "licon/nn/node/neuron.hpp"

namespace licon
{

namespace nn
{

class Relu : public NeuronNode<F> {

public:
    static std::unique_ptr<OpNode<F> > CreateRelu(F negative_slope = 0.2);

    // the name
    virtual inline std::string name() const { return "relu"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);


protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data; //when forwad save, backward remove

    F m_negative_slope;

private:
    Relu(F negative_slope);
};

} //nn

} //licon



#endif /*LICON_RELU_NODE_HPP_*/