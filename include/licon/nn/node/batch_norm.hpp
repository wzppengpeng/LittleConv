#ifndef LICON_NN_NODE_BATCH_NORM_HPP
#define LICON_NN_NODE_BATCH_NORM_HPP

/**
 * the batch norm layer, will have scale layer in place
 */
#include <unordered_map>


#include "licon/nn/node/neuron.hpp"

namespace licon
{

namespace nn
{

// the batchnorm node
class BatchNorm : public NeuronNode<F> {

public:
    // the heap create functions
    static nn::NodePtr CreateBatchNorm(int num_features, F eps = 1e-5, F momentum = 0.1, bool affine = true);

    // the node's name(type)
    virtual inline std::string name() const { return "batch_norm"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

    // init params
    virtual void InitParameters();

    // register parameters
    virtual std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > RegisterWeights();

    // get the weight and bias point
    virtual inline utils::ETensor<F>* weight() { return &m_weights; }
    virtual inline utils::ETensor<F>* bias() { return &m_bias; }

    // the state dict getter and loader
    virtual std::unordered_map<std::string, utils::ETensor<F>* > StateDict();
    virtual void LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data = nullptr;
    // the post saved input grad
    utils::ETensor<F>* m_top_grad = nullptr;

    // the constructor parameters
    int m_in_channels; // should be same with numfeatures
    F m_eps;
    F m_momentum;
    bool m_affine;


    // the weight and bias
    utils::ETensor<F> m_weights; //inchanels * 1
    utils::ETensor<F> m_bias; //1 * inchanels

    // save the weights grad
    utils::ETensor<F> m_weights_grad;
    utils::ETensor<F> m_bias_grad;

    // have been init
    bool m_has_initialzed = false;


private:
    BatchNorm(int num_features, F eps, F momentum, bool affine);

    // some temp varibales for inner save
    int m_spatial_dim;

    // the mean and variance
    utils::ETensor<F> m_mean;
    utils::ETensor<F> m_mean_current;
    utils::ETensor<F> m_variance;
    utils::ETensor<F> m_variance_current;
    utils::ETensor<F> m_stddev; //the norm std fenmu

    // the temp output of batchnorm for not scale
    utils::ETensor<F>* m_Y = nullptr;
    utils::ETensor<F> m_xhat;
    utils::ETensor<F> m_xhat_grad;


protected:
    // the init function
    void Init();

    // the update function for the mean and variance
    inline void PostUpdate();

    inline void CalculateStddev(const utils::ETensor<F>& variance);

    // the forward of scale
    void ForwardScale();

    // the backward of scale
    void BackwardScale();

    // reshape functions
    // forward reshape the data
    void ReshapeForward(const std::vector<utils::ETensor<F>* >& bottom);
    // backward reshape the grad
    void ReshapeBackward(const std::vector<utils::ETensor<F>* >& top);

    inline bool use_global_stats() const { return this->m_phase == Phase::TEST; }

};

} //nn

} //licon


#endif /*LICON_NN_NODE_BATCH_NORM_HPP*/