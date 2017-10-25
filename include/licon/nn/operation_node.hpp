#ifndef LICON_OPERATION_NODE_HPP_
#define LICON_OPERATION_NODE_HPP_

#include <cassert>

#include <string>
#include <memory>

#include <unordered_map>

#include "licon/common.hpp"
#include "licon/utils/etensor.hpp"

/**
 * the base interface of op node
 * will have the data output and grad output
 * the base interface forward and backward
 */

namespace licon
{

namespace nn
{

template<typename Dtype>
class OpNode {
public:
    // the basic constructor is not needed
    OpNode() : m_phase(Phase::TRAIN) {}
    // virtual deconstrcutor
    virtual ~OpNode() {}

    // the set up interface, may be not used
    virtual void SetUp() {} //ex: set the num of connection vectors

    /**
     * Setters
     */
    // set the phase
    virtual inline void set_phase(Phase phase) { m_phase = phase; }
    // set the former node
    virtual inline void set_former_node(OpNode<Dtype>* former, size_t idx = 0) {
        assert(m_former_nodes.size() > idx);
        m_former_nodes[idx] = former;
    }
    // set the after node
    virtual inline void set_after_node(OpNode<Dtype>* after, size_t idx = 0) {
        assert(m_after_nodes.size() > idx);
        m_after_nodes[idx] = after;
    }
    // add the former node into the connection(may not use)
    virtual inline void add_former_node(OpNode<Dtype>* former) { m_former_nodes.emplace_back(former); }
    // add the after node into the connection
    virtual inline void add_after_node(OpNode<Dtype>* after) { m_after_nodes.emplace_back(after); }
    // set the node name
    virtual void set_node_name(std::string node_name) { m_node_name = std::move(node_name); }

    /**
     * Getters
     */
    // get the exact num of op node's before, if num > 0, then need to check
    virtual inline int exact_num_former_nodes() const { return -1; }
    // get the exact num of op node's after
    virtual inline int exact_num_after_nodes() const { return -1; }
    // get the name of this op node
    virtual inline std::string name() const { return ""; }
    inline std::string node_name() const { return m_node_name; }
    // get this node's output data
    virtual inline utils::ETensor<Dtype>* data() { return &m_data; }
    // get this node's output grad
    virtual inline utils::ETensor<Dtype>* grad() { return &m_grad; }

    // register parameters interface(TODO:)
    // TODO:
    // the parameter init interface
    virtual void InitParameters() {} //when graph is built calling
    // register the weight and its weight grad tensor
    virtual std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > RegisterWeights() { return {}; };

    // the weight and bias state dict getter(node_name_weight->weight, node_name_bias->bias)
    virtual std::unordered_map<std::string, utils::ETensor<F>* > StateDict() { return {}; };
    // the weight and bias state dict loader
    virtual void LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict) {}


    // get the weight for init
    virtual inline utils::ETensor<Dtype>* weight() { return nullptr; }
    virtual inline utils::ETensor<Dtype>* bias() { return nullptr; }



    // the forward interface
    // compute the data output of this op node
    // bottom is the former op nodes' output, may be need multi operation datas
    virtual void Forward(const std::vector<utils::ETensor<Dtype>* >& bottom) = 0;

    // the backward interface
    // compute the grad output of thos op node
    // top is the after op nodes' grad
    virtual void Backward(const std::vector<utils::ETensor<Dtype>* >& top) = 0;

    // the container interface, only the container node need to overwrite
    // this is the stack opnode interface
    virtual void Add(std::unique_ptr<OpNode<F> > neuron_node) {}
    // add a neuron node, give its sub node name at the same time
    virtual void Add(std::string sub_node_name, std::unique_ptr<OpNode<F> > neuron_node) {}
    // to get the node in the stack container
    virtual inline OpNode<Dtype>* operator[] (size_t index) { return index == 0 ? this : nullptr;  }
    virtual inline const OpNode<Dtype>* operator[] (size_t index) const { return index == 0 ? this : nullptr; }

protected:
    // the personal members
    Phase m_phase; //train or test

    // the parameters for different op_node is different

    // the data output
    utils::ETensor<Dtype> m_data;

    // the grad output
    utils::ETensor<Dtype> m_grad; //the shape should be same with the input data

    // the nodes connect before
    std::vector<OpNode<Dtype>* > m_former_nodes;

    // the nodes connect after
    std::vector<OpNode<Dtype>* > m_after_nodes;

    // the name of this node(for save model and load model)
    std::string m_node_name;

    // private functions
    // the function to step update the parameters
    virtual void Step() {}; //may be not good enuough

    // the check interface(inner function) to check if the count is match
    virtual inline void check_forward(const std::vector<utils::ETensor<Dtype>* >& bottom) {
        if(exact_num_former_nodes() > 0) { ASSERT(bottom.size() == m_former_nodes.size(), "The forward size mismatch", name()); }
    }
    virtual inline void check_backward(const std::vector<utils::ETensor<Dtype>* >& top) {
        if(exact_num_after_nodes() > 0 ) { ASSERT(top.size() == m_after_nodes.size(), "The backward size miss match", name()); }
    }

};

// the opnode ptr alias
typedef std::unique_ptr<OpNode<F> > Model;
typedef std::unique_ptr<OpNode<F> > NodePtr;


} //nn

} //licon


#endif /*LICON_OPERATION_NODE_HPP_*/