#ifndef LICON_NN_DROPUOUT_HPP_
#define LICON_NN_DROPUOUT_HPP_

#include "licon/nn/node/neuron.hpp"

/**
 * the dropout node
 */
namespace licon
{

namespace nn
{

class Dropout : public NeuronNode<F> {

public:
    static std::unique_ptr<OpNode<F> > CreateDropout(F p);

    // the name
    virtual inline std::string name() const { return "dropout"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    // the threshold
    F m_threshold;
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data; //when forwad save, backward remove

    // the mask of rand to get if use train phase
    utils::ETensor<unsigned int> m_mask;

    // the scale of undropped input to make the distrubution ok
    F m_scale;

private:
    Dropout(F p);

};

} //nn

} //licon

#endif /*LICON_NN_DROPUOUT_HPP_*/