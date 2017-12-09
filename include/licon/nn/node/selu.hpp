#ifndef LICON_NN_NODE_SELU_HPP_
#define LICON_NN_NODE_SELU_HPP_

/**
 * the selu activation, base on elu
 */

#include "licon/nn/node/neuron.hpp"


namespace licon
{

namespace nn
{

class SELU : public NeuronNode<F> {

public:
    // the heap creator
    static nn::NodePtr CreateSELU();

    // the name
    virtual inline std::string name() const { return "selu"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data; //saved for backward

    // the paramethers
    F m_scale;
    F m_alpha;


private:
    SELU();

};

} //nn

} //licon


#endif /*LICON_NN_NODE_SELU_HPP_*/