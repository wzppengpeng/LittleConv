/**
 * the container details of neuron squential node
 */
#include "licon/nn/node/neuron_squential.hpp"


using namespace std;

namespace licon
{

namespace nn
{

std::unordered_map<std::string, utils::ETensor<F>* > Squential::StateDict() {
    CHECK(m_node_name.empty() == false);
    std::unordered_map<std::string, utils::ETensor<F>* > state_dict_squtienial;
    for(auto& node : m_nodes) {
        auto state_dict = node->StateDict();
        for(auto& p : state_dict) {
            state_dict_squtienial.emplace(p.first, p.second);
        }
    }
    return state_dict_squtienial;
}

void Squential::LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict) {
    CHECK(m_node_name.empty() == false);
    for(auto& node : m_nodes) {
        node->LoadStateDict(state_dict);
    }
}

std::unique_ptr<OpNode<F> > Squential::CreateSquential() {
    return std::unique_ptr<OpNode<F> >(new Squential());
}

void Squential::set_node_name(std::string node_name) {
    m_node_name = std::move(node_name);
    for(size_t i = 0; i < size(); ++i) {
        m_nodes[i]->set_node_name(m_node_name + "_" + m_nodes[i]->name() + "_" + std::to_string(i));
    }
}

void Squential::Add(std::unique_ptr<OpNode<F> > neuron_node) {
    // first add the new neuron node into the vector container
    m_nodes.emplace_back(std::move(neuron_node));
    // connect the former and after node
    if(m_nodes.size() > 1) {
        size_t be = m_nodes.size() - 2;
        m_nodes[be]->set_after_node(m_nodes.back().get());
        m_nodes.back()->set_former_node(m_nodes[be].get());
    }
}

void Squential::Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node) {
    Add(std::move(neuron_node));
    m_nodes.back()->set_node_name(sub_node_name);
}

void Squential::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    utils::ETensor<F>* input = bottom[0];
    for(auto& node : m_nodes) {
        node->Forward({input});
        input = node->data();
    }
}

void Squential::Backward(const std::vector<utils::ETensor<F>* >& top) {
    check_backward(top);
    utils::ETensor<F>* input = top[0];
    for(auto it = m_nodes.rbegin(); it != m_nodes.rend(); ++it) {
        (*it)->Backward({input});
        input = (*it)->grad();
    }
}

void Squential::InitParameters() {
    for(auto& node : m_nodes) {
        node->InitParameters();
    }
}

std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > Squential::RegisterWeights() {
    std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > weights_squential;
    for(auto& node : m_nodes) {
        auto sub_node_weights = node->RegisterWeights();
        for(auto& w_pair : sub_node_weights) weights_squential.emplace_back(std::move(w_pair));
    }
    return std::move(weights_squential);
}

Squential::Squential()
    : NeuronNode<F>(),
    m_has_initialzed(false)
{
    NeuronNode<F>::Resize();
}

} //nn

} //licon