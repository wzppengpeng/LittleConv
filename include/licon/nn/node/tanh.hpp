#ifndef LICON_TANH_NODE_HPP_
#define LICON_TANH_NODE_HPP_

/**
 * the tanh active node(-1 ~ 1)
 */

#include "licon/nn/node/neuron.hpp"

namespace licon
{

namespace nn
{

class Tanh : public NeuronNode<F> {
public:
    // the heap create function
    static std::unique_ptr<OpNode<F> > CreateTanh();

    // the name of this node
    virtual inline std::string name() const { return "tanh"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    bool m_has_forwarded;

private:
    Tanh();


};

} //nn

} //licon

#endif /*LICON_TANH_NODE_HPP_*/