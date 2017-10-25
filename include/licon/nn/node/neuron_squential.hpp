#ifndef LICON_NN_SQUENTIAL_HPP_
#define LICON_NN_SQUENTIAL_HPP_

/**
 * the neuron squential, one by one
 */
#include "licon/nn/node/neuron.hpp"

namespace licon
{

namespace nn
{

class Squential : public NeuronNode<F> {

public:
    // the heap create function
    static std::unique_ptr<OpNode<F> > CreateSquential();

    // the name of this node
    virtual inline std::string name() const { return "squential"; }

    // set the name of the squential node
    virtual void set_node_name(std::string node_name);

    // set the train or test phase
    virtual inline void set_phase(Phase phase) {
        OpNode<F>::set_phase(phase);
        for(auto& node : m_nodes) {
            node->set_phase(phase);
        }
    }

    // add a neuron node into the squential container
    virtual void Add(std::unique_ptr<OpNode<F> > neuron_node);
    // add a neuron node, give its sub node name at the same time
    virtual void Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node);

    virtual inline void set_former_node(OpNode<F>* former, size_t idx = 0) {
        OpNode<F>::set_former_node(former, idx);
        if(!empty()) { m_nodes.front()->set_former_node(former, idx); }
    }

    virtual inline void set_after_node(OpNode<F>* after, size_t idx = 0) {
        OpNode<F>::set_after_node(after, idx);
        if(!empty()) { m_nodes.back()->set_after_node(after, idx); }
    }

    // get the data and grad
    virtual inline utils::ETensor<F>* data() {
        if(m_nodes.empty()) return OpNode<F>::data();
        else return m_nodes.back()->data();
    }

    virtual inline utils::ETensor<F>* grad() {
        if(m_nodes.empty()) return OpNode<F>::grad();
        else return m_nodes.front()->grad();
    }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

    // init params
    virtual void InitParameters();

    // register parameters
    virtual std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > RegisterWeights();

    // get the operation node
    virtual inline OpNode<F>* operator[] (size_t index) { return m_nodes[index].get(); }
    virtual inline const OpNode<F>* operator[] (size_t index) const { return m_nodes[index].get(); }

    // get thelength of the squential node
    inline size_t size() const { return m_nodes.size(); }
    inline bool empty() const { return m_nodes.empty(); }

    // the state dict getter and loader
    virtual std::unordered_map<std::string, utils::ETensor<F>* > StateDict();
    virtual void LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict);


private:
    // the constructor
    Squential();

    bool m_has_initialzed;

    // the container of neuron nodes
    std::vector<std::unique_ptr<OpNode<F> > > m_nodes;



};

} //nn

} //licon


#endif /*LICON_NN_SQUENTIAL_HPP_*/