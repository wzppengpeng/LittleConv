#include "licon/nn/node/elt_wise_sum.hpp"


#include "licon/nn/functor/elt_wise_sum.hpp"
#include "licon/nn/functor/path.hpp"

using namespace std;


namespace licon
{

namespace nn
{

nn::NodePtr EltWiseSum::CreateEltWiseSum(bool need_self) {
    return NodePtr(new EltWiseSum(need_self));
}

void EltWiseSum::Add(std::unique_ptr<OpNode<F> > neuron_node) {
    m_nodes.emplace_back(std::move(neuron_node));
}

void EltWiseSum::Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node) {
    Add(std::move(neuron_node));
    m_nodes.back()->set_node_name(std::move(sub_node_name));
}

void EltWiseSum::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    // use path way to get different copy of input data
    ASSERT(empty() == false, "The EltWiseSum Container is empty");
    for(auto& node : m_nodes) {
        node->Forward(bottom);
    }
    // sum all output together
    vector<utils::ETensor<F>* > sub_out;
    if(m_need_self) sub_out.emplace_back(bottom[0]);
    for(auto& node : m_nodes) {
        sub_out.emplace_back(node->data());
    }
    this->m_data = std::move(EltWiseSumFunctor()(sub_out)[0]);
}

void EltWiseSum::Backward(const std::vector<utils::ETensor<F>* >& top) {
    check_backward(top);
    ASSERT(empty() == false, "The EltWiseSum Container is empty");
    // output the grad to each sub node
    for(auto& node : m_nodes) {
        node->Backward(top);
    }
    vector<utils::ETensor<F>* > sub_grad;
    if(m_need_self) sub_grad.emplace_back(top[0]);
    for(auto& node : m_nodes) {
        sub_grad.emplace_back(node->grad());
    }
    this->m_grad = std::move(PathFunctor(sub_grad.size()).Backward(sub_grad)[0]);
}


EltWiseSum::EltWiseSum(bool need_self) : Stack<F>(), m_need_self(need_self) {}

} //nn

} //licon