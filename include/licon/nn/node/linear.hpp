#ifndef LICON_NN_LINEAR_HPP_
#define LICON_NN_LINEAR_HPP_

/**
 * the fc layer
 */
#include "licon/nn/node/neuron.hpp"


namespace licon
{

namespace nn
{

class Linear : public NeuronNode<F> {
public:
    static std::unique_ptr<OpNode<F> > CreateLinear(int input, int output);

    // the name
    virtual inline std::string name() const { return "linear"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

    // init params
    virtual void InitParameters();

    // register parameters
    virtual std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > RegisterWeights();

    // the specaial init params
    void InitWeights(F mean, F var);
    void InitBias(F val);

    // get the weight and bias point
    virtual inline utils::ETensor<F>* weight() { return &m_weights; }
    virtual inline utils::ETensor<F>* bias() { return &m_bias; }

    // the state dict getter and loader
    virtual std::unordered_map<std::string, utils::ETensor<F>* > StateDict();
    virtual void LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data; //when forwad save, backward remove


private:
    // the constructor
    Linear(int input, int output);

    // the shapes
    int m_input_dim;
    int m_output_dim;

    bool m_has_initialzed;

    // the parameters
    utils::ETensor<F> m_weights; //input * output
    utils::ETensor<F> m_bias; //1 * output

    // save the weights grad
    utils::ETensor<F> m_weights_grad;
    utils::ETensor<F> m_bias_grad;


};

} //nn

} //licon


#endif /*LICON_NN_LINEAR_HPP_*/