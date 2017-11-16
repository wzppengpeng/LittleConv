#ifndef LICON_NN_NODE_ELU_HPP
#define LICON_NN_NODE_ELU_HPP

/**
 * the elu activations
 */
#include "licon/nn/node/neuron.hpp"


namespace licon
{

namespace nn
{

class ELU : public NeuronNode<F> {

public:
    // the heap create function
    static nn::NodePtr CreateELU(F alpha = 1.0);

    // the name
    virtual inline std::string name() const { return "elu"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data; //saved for backward

    F m_alpha;

private:
    // constructor
    ELU(F alpha);

};

} //nn

} //licon

#endif /*LICON_NN_NODE_ELU_HPP*/