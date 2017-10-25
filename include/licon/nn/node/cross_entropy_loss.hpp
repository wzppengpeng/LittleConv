#ifndef LICON_NN_CROSS_ENTROPY_LOSS_HPP
#define LICON_NN_CROSS_ENTROPY_LOSS_HPP

#include "licon/nn/node/loss.hpp"

/**
 * the negative log loss
 */

namespace licon
{

namespace nn
{

/**
 * the output will be a scalar
 */

class CrossEntropyLoss : public LossNode<F> {

public:
    // the heap create function
    static std::unique_ptr<OpNode<F> > CreateCrossEntropyLoss(bool size_average = true);

    // the name
    virtual inline std::string name() const { return "cross_entropy_loss"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

    // get the exact num of op node's before, if num > 0, then need to check
    virtual inline int exact_num_former_nodes() const { return 2; }

protected:
    // if need size average
    bool m_size_average;

    // the pre saved input data
    utils::ETensor<F>* m_bottom_data;
    utils::ETensor<F>* m_bottom_label;


private:
    CrossEntropyLoss(bool size_average);

};

} //nn

} //licon


#endif /*LICON_NN_CROSS_ENTROPY_LOSS_HPP*/