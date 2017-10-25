#ifndef LICON_SIGMOID_NODE_HPP_
#define LICON_SIGMOID_NODE_HPP_

/**
 * the sigmoid active node
 */
#include "licon/nn/node/neuron.hpp"

namespace licon
{

namespace nn
{

class Sigmoid : public NeuronNode<F> {

public:
    // the head create function
    static std::unique_ptr<OpNode<F> > CreateSigmoid();

    // the name of this node
    virtual inline std::string name() const { return "sigmoid"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);


protected:
    bool m_has_forwarded;

private:
    Sigmoid();

};

} //nn

} //licon

#endif /*LICON_SIGMOID_NODE_HPP_*/