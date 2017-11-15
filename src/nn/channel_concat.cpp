#include "licon/nn/node/channel_concat.hpp"


#include "licon/nn/functor/path.hpp"


using namespace std;

namespace licon
{

namespace nn
{

nn::NodePtr ChannelConcat::CreateChannelConcat() {
    return nn::NodePtr(new ChannelConcat());
}

void ChannelConcat::Add(std::unique_ptr<OpNode<F> > neuron_node) {
    m_nodes.emplace_back(std::move(neuron_node));
}

void ChannelConcat::Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node) {
    Add(std::move(neuron_node));
    m_nodes.back()->set_node_name(std::move(sub_node_name));
}

void ChannelConcat::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    ASSERT(empty() == false, "The ChannelConcat Container is empty");
    for(auto& node : m_nodes) {
        node->Forward(bottom);
    }
    // channel concat the subnodes
    vector<utils::ETensor<F>* > subnodes_outputs;
    for(auto& node : m_nodes) subnodes_outputs.emplace_back(node->data());
    this->m_data = std::move(m_channel_concat_functor(subnodes_outputs).front());
}

void ChannelConcat::Backward(const std::vector<utils::ETensor<F>* >& top) {
    check_backward(top);
    ASSERT(empty() == false, "The ChannelConcat Container is empty");
    auto sub_grads = m_channel_concat_functor.Backward(top);
    for(size_t i = 0; i < size(); ++i) {
        m_nodes[i]->Backward({&sub_grads[i]});
    }
    vector<utils::ETensor<F>* > sub_grads_outouts;
    for(auto& node : m_nodes) sub_grads_outouts.emplace_back(node->grad());
    this->m_grad = std::move(PathFunctor(sub_grads_outouts.size()).Backward(sub_grads_outouts)[0]);
}

ChannelConcat::ChannelConcat() : Stack<F>()
{
}

} //nn

} //licon