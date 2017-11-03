#ifndef LICON_NN_NODE_NEURON_STACK_HPP_
#define LICON_NN_NODE_NEURON_STACK_HPP_

/**
 * the container of stack opnode(squential and multi way combine)
 */
#include "licon/nn/node/neuron.hpp"


namespace licon
{

namespace nn
{

template<typename Dtype>
class Stack : public NeuronNode<Dtype> {

public:
    Stack() : NeuronNode<F>() {}

    // set the node name one by one
    virtual inline void set_node_name(std::string node_name) {
        this->m_node_name = std::move(node_name);
        for(size_t i = 0; i < size(); ++i) {
            m_nodes[i]->set_node_name(this->m_node_name + "_" + m_nodes[i]->name() + "_" + std::to_string(i));
        }
    }

    // set the train or test phase
    virtual inline void set_phase(Phase phase) {
        OpNode<Dtype>::set_phase(phase);
        for(auto& node : m_nodes) {
            node->set_phase(phase);
        }
    }

    // the add function for different stack container
    virtual void Add(std::unique_ptr<OpNode<Dtype> > neuron_node) = 0;
    // add a neuron node, give its sub node name at the same time
    virtual void Add(std::string sub_node_name, std::unique_ptr<OpNode<Dtype> > neuron_node) = 0;

    // init params, cause the container most probably have parameters
    virtual void InitParameters() {
        for(auto& node : m_nodes) {
            node->InitParameters();
        }
    }

    // register all the parameters in the container
    virtual std::vector<std::pair<utils::ETensor<Dtype>*, utils::ETensor<Dtype>* > > RegisterWeights() {
        std::vector<std::pair<utils::ETensor<Dtype>*, utils::ETensor<Dtype>* > > weights_stack;
        for(auto& node : m_nodes) {
            auto sub_node_weights = node->RegisterWeights();
            for(auto& w_pair : sub_node_weights) weights_stack.emplace_back(std::move(w_pair));
        }
        return std::move(weights_stack);
    }


    // get the operation node
    virtual inline OpNode<Dtype>* operator[] (size_t index) { return m_nodes[index].get(); }
    virtual inline const OpNode<Dtype>* operator[] (size_t index) const { return m_nodes[index].get(); }

    // get thelength of the squential node
    virtual inline size_t size() const { return m_nodes.size(); }
    virtual inline bool empty() const { return m_nodes.empty(); }

    // get the state dict one by one
    virtual std::unordered_map<std::string, utils::ETensor<Dtype>* > StateDict() {
        CHECK(this->m_node_name.empty() == false);
        std::unordered_map<std::string, utils::ETensor<Dtype>* > state_dict_stack;
        for(auto& node : m_nodes) {
            auto state_dict = node->StateDict();
            for(auto& p : state_dict) {
                state_dict_stack.emplace(p.first, p.second);
            }
        }
        return state_dict_stack;
    }

    // load the stadict one by one
    virtual void LoadStateDict(const std::unordered_map<std::string, utils::ETensor<Dtype> >& state_dict) {
        CHECK(this->m_node_name.empty() == false);
        for(auto& node : m_nodes) {
            node->LoadStateDict(state_dict);
        }
    }


protected:
    bool m_has_initialzed = false;
    // the detail part of the nodes in the stack container
    std::vector<std::unique_ptr<OpNode<Dtype> > > m_nodes;

};

} //nn

} //licon






#endif /*LICON_NN_NODE_NEURON_STACK_HPP_*/