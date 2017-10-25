#ifndef LICON_NN_SOFTMAX_HPP_
#define LICON_NN_SOFTMAX_HPP_

/**
 * the softmax activetion
 */
#include "licon/nn/node/neuron.hpp"

namespace licon
{

namespace nn
{

class Softmax : public NeuronNode<F> {

public:
    static std::unique_ptr<OpNode<F> > CreateSoftmax();

    // the name
    virtual inline std::string name() const { return "softmax"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data; //when forwad save, backward remove

private:
    Softmax();

};

} //nn

} //licon

#endif /*LICON_NN_SOFTMAX_HPP_*/