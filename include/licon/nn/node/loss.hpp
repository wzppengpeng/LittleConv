#ifndef LICON_NN_LOSS_HPP_
#define LICON_NN_LOSS_HPP_

#include "licon/nn/operation_node.hpp"

namespace licon
{

namespace nn
{

// the loss node only have one after node
template<typename Dtype>
class LossNode : public OpNode<Dtype> {
public:
    // constructor
    LossNode() : OpNode<Dtype>() {
        Resize();
    }

    // deconstrcutor
    virtual ~LossNode() {}

    // the nums of post node
    virtual inline int exact_num_after_nodes() const  { return 1; }

protected:
    virtual inline void Resize() {
        OpNode<Dtype>::m_after_nodes.resize(1, nullptr);
    }

};

} //nn

} //licon


#endif /*LICON_NN_LOSS_HPP_*/