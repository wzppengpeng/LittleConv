#ifndef LICON_NN_NODE_ELT_WISE_SUM_HPP
#define LICON_NN_NODE_ELT_WISE_SUM_HPP

/**
 * the container of hstack some opnode together then element wise sum into one tensor
 */
#include "licon/nn/node/neuron_stack.hpp"



namespace licon
{

namespace nn
{

class EltWiseSum : public Stack<F> {

public:
    // the heap creator, if set true, the container will set one path to self
    static nn::NodePtr CreateEltWiseSum(bool need_self = true);

    // the name of this node
    virtual inline std::string name() const { return "elt_wise_sum"; }

    // add a neuron node into the squential container
    virtual void Add(std::unique_ptr<OpNode<F> > neuron_node);
    // add a neuron node, give its sub node name at the same time
    virtual void Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node);

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

private:
    // the constructor
    EltWiseSum(bool need_self);

    bool m_need_self;

};

} //nn

} //licon



#endif /*LICON_NN_NODE_ELT_WISE_SUM_HPP*/