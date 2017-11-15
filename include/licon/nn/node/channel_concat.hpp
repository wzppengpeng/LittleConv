#ifndef LICON_NN_NODE_CHANNEL_CONCAT_HPP_
#define LICON_NN_NODE_CHANNEL_CONCAT_HPP_

/**
 * the container to hstack opnodes together then channel concat them into one tensor
 */
#include "licon/nn/node/neuron_stack.hpp"

#include "licon/nn/functor/channel_concat.hpp"


namespace licon
{

namespace nn
{

class ChannelConcat : public Stack<F> {
public:
    // the heap creater function
    static nn::NodePtr CreateChannelConcat();

    // the name of this node
    virtual inline std::string name() const { return "channel_concat"; }

    // add a neuron node into the squential container
    virtual void Add(std::unique_ptr<OpNode<F> > neuron_node);
    // add a neuron node, give its sub node name at the same time
    virtual void Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node);

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

private:
    // the constructor
    ChannelConcat();

    // the channel concat functor
    ChanelConcatFunctor m_channel_concat_functor;


};

} //nn

} //licon


#endif /*LICON_NN_NODE_CHANNEL_CONCAT_HPP_*/