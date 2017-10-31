#include "licon/nn/node/batch_norm.hpp"

#include "licon/utils/etensor_args.hpp"
#include "licon/nn/init.hpp"

#include "function/help_function.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

using wzp::print;

// global area
const static F RANGE_MIN = -1.;
const static F RANGE_MAX = 1.;
const static F BIAS_VAL = 0.;


namespace licon
{

namespace nn
{

nn::NodePtr BatchNorm::CreateBatchNorm(int num_features, F eps, F momentum, bool affine) {
    return nn::NodePtr(new BatchNorm(num_features, eps, momentum, affine));
}

BatchNorm::BatchNorm(int num_features, F eps, F momentum, bool affine)
            : m_in_channels(num_features), m_eps(eps), m_momentum(momentum), m_affine(affine)
{
    Init();
}

void BatchNorm::Init() {
    NeuronNode<F>::Resize();
    // resize some tensors
    m_mean.Reshape(m_in_channels, 1, 1, 1);
    utils::ETensorArgs<F>::fill(m_mean, 0.);
    m_mean_current.Reshape(m_in_channels, 1, 1, 1);
    utils::ETensorArgs<F>::fill(m_mean_current, 0.);
    m_variance.Reshape(m_in_channels, 1, 1, 1);
    utils::ETensorArgs<F>::fill(m_variance, 0.);
    m_variance_current.Reshape(m_in_channels, 1, 1, 1);
    utils::ETensorArgs<F>::fill(m_variance_current, 0.);
    m_stddev.Reshape(m_in_channels, 1, 1, 1);
    utils::ETensorArgs<F>::fill(m_stddev, 0.);
    if(m_affine) {
        m_weights.Reshape(m_in_channels, 1, 1, 1);
        m_weights_grad.Reshape(m_in_channels, 1, 1, 1);
        m_bias.Reshape(1, m_in_channels, 1, 1);
        m_bias_grad.Reshape(1, m_in_channels, 1, 1);
        InitParameters();
    }
}

inline void BatchNorm::PostUpdate() {
    for(int i = 0; i < m_in_channels; ++i) {
        m_mean(i) = m_mean(i) * m_momentum + (1. - m_momentum) * m_mean_current(i);
        m_variance(i) = m_variance(i) * m_momentum + (1. - m_momentum) * m_variance_current(i);
    }
}

inline void BatchNorm::CalculateStddev(const utils::ETensor<F>& variance) {
    for(int i = 0; i < m_in_channels; ++i) {
        m_stddev(i) = sqrt(variance(i) + m_eps);
    }
}

void BatchNorm::ForwardScale() {
    if(m_affine) {
        wzp::ParallelRange(m_data.num(), [this](int n) {
            F* data_ptr = m_data.mutable_ptr(n);
            const F* xhat_ptr = m_xhat.ptr(n);
            const F* weight_ptr = m_weights.ptr();
            const F* bias_ptr = m_bias.ptr();
            for(int j = 0; j < m_in_channels; ++j) {
                for(int k = 0; k < m_spatial_dim; ++k) {
                    *data_ptr++ = (*xhat_ptr++) * weight_ptr[j] + bias_ptr[j];
                }
            }
        });
    }
}

void BatchNorm::BackwardScale() {
    if(m_affine) {
        // get the grad for weight
        utils::ETensor<F> buffer(m_top_grad->shape());
        const F* dy_ptr = m_top_grad->ptr();
        const F* xhat_ptr = m_xhat.ptr();
        F* buf_ptr = buffer.mutable_ptr();
        for(size_t i = 0; i < m_top_grad->count(); ++i) {
            buf_ptr[i] = dy_ptr[i] * xhat_ptr[i];
        }
        // reset the weight grad to zero
        utils::ETensorArgs<F>::fill(m_weights_grad, 0.);
        utils::ETensorArgs<F>::fill(m_bias_grad, 0.);
        // get the weight grad
        utils::ETensorArgs<F>::channel_sum(buffer, m_weights_grad.mutable_ptr());
        // get the bias grad
        utils::ETensorArgs<F>::channel_sum(*m_top_grad, m_bias_grad.mutable_ptr());
        // compute the xhat grad
        wzp::ParallelRange(m_xhat.num(), [this](int n) {
            F* grad_ptr = m_xhat_grad.mutable_ptr(n);
            const F* dy_ptr = m_top_grad->ptr(n);
            const F* weight_ptr = m_weights.ptr();
            for(int j = 0; j < m_in_channels; ++j) {
                for(int k = 0; k < m_spatial_dim; ++k) {
                    *grad_ptr++ = (*dy_ptr++) * weight_ptr[j];
                }
            }
        });
    }
}

void BatchNorm::ReshapeForward(const std::vector<utils::ETensor<F>* >& bottom) {
    ASSERT(bottom[0]->channel() == m_in_channels, "The Input Data Chanel Mismatch", m_in_channels);
    m_bottom_data = bottom[0];
    // reshape the m_data and m_xhat
    m_data.Reshape(m_bottom_data->shape());
    if(m_affine) {
        m_xhat.Reshape(m_bottom_data->shape());
        m_Y = &m_xhat;
    } else {
        m_Y = &m_data;
    }
    m_spatial_dim = m_bottom_data->count(2);
}

void BatchNorm::ReshapeBackward(const std::vector<utils::ETensor<F>* >& top) {
    ASSERT(top[0]->num() == m_data.num() && top[0]->channel() == m_data.channel()
        && top[0]->height() == m_data.height() && top[0]->width() == m_data.width(),
        "Shape Miss Match");
    if(m_affine) m_xhat_grad.Reshape(m_bottom_data->shape());
    m_grad.Reshape(m_bottom_data->shape());
    m_top_grad = top[0];
}

void BatchNorm::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    ReshapeForward(bottom);
    // first calculate mean and variance
    utils::ETensor<F>& mean = use_global_stats() ? m_mean : m_mean_current;
    utils::ETensor<F>& variance = use_global_stats() ? m_variance : m_variance_current;
    if(!use_global_stats()) {
        utils::ETensorArgs<F>::channel_variance(*m_bottom_data, mean.mutable_ptr(), variance.mutable_ptr());
    }
    // calculate the stddev
    CalculateStddev(variance);
    // normalize
    // y = (x - mean) ./ sqrt(variance + eps)
    // save in the xhat
    wzp::ParallelRange(m_bottom_data->num(), [this, &mean](int n) {
        const F* bottom_data_ptr = m_bottom_data->ptr(n);
        F* data_ptr = m_Y->mutable_ptr(n);
        for(int j = 0; j < m_in_channels; ++j) {
            F m = mean(j);
            for(int k = 0; k < m_spatial_dim; ++k) {
                *data_ptr++ = (*bottom_data_ptr++ - m) / m_stddev(j);
            }
        }
    });
    // save the mean and varance
    if(!use_global_stats()) PostUpdate();
    ForwardScale();
}

void BatchNorm::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ReshapeBackward(top);
    BackwardScale(); //get the grad in xhat_grad
    auto& prev_delta = m_grad;
    auto& curr_delta = m_affine ? m_xhat_grad : *m_top_grad;
    const auto& curr_out = m_data;
    int num_samples = curr_out.num();
    utils::ETensor<F> delta_dot_y = curr_out;
    std::vector<F> mean_delta_dot_y, mean_delta;
    // copy the delta dot y double
    F* delta_dot_y_ptr = delta_dot_y.mutable_ptr();
    const F* curr_delta_ptr = curr_delta.ptr();
    for(size_t i = 0; i < m_data.count(); ++i) {
        delta_dot_y_ptr[i] *= curr_delta_ptr[i];
    }
    // get the mean of delta y and mean of delta y dot dy
    utils::ETensorArgs<F>::channel_mean(delta_dot_y, mean_delta_dot_y);
    utils::ETensorArgs<F>::channel_mean(curr_delta, mean_delta);
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    wzp::ParallelRange(num_samples, [&](int n) {
        F* prev_delta_ptr_n = prev_delta.mutable_ptr(n);
        const F* curr_delta_ptr_n = curr_delta.ptr(n);
        const F* curr_out_ptr_n = curr_out.ptr(n);
        for(int j = 0; j < m_in_channels; ++j) {
            for(int k = 0; k < m_spatial_dim; ++k) {
                *prev_delta_ptr_n = (*curr_delta_ptr_n++) - mean_delta[j] -
                                    mean_delta_dot_y[j] * (*curr_out_ptr_n++);
                *prev_delta_ptr_n /= m_stddev(j);
                ++prev_delta_ptr_n;
            }
        }
    });
    m_bottom_data = nullptr;
}

void BatchNorm::InitParameters() {
    if(m_has_initialzed == false) {
        nn::ParamInit<F>::uniform(m_weights, RANGE_MIN, RANGE_MAX);
        nn::ParamInit<F>::constant(m_bias, BIAS_VAL);
    }
    m_has_initialzed = true;
}

std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > BatchNorm::RegisterWeights() {
    return {{&m_weights, &m_weights_grad}, {&m_bias, &m_bias_grad}};;
}

std::unordered_map<std::string, utils::ETensor<F>* > BatchNorm::StateDict() {
    CHECK(m_node_name.empty() == false);
    std::unordered_map<std::string, utils::ETensor<F>* > state_dict_bn;
    state_dict_bn.emplace(m_node_name + "MEAN", &m_mean);
    state_dict_bn.emplace(m_node_name + "VARIANCE", &m_variance);
    if(m_affine) {
        state_dict_bn.emplace(m_node_name + "W", weight());
        state_dict_bn.emplace(m_node_name + "B", bias());
    }
    return state_dict_bn;
}

void BatchNorm::LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict) {
    CHECK(m_node_name.empty() == false);
    ASSERT(state_dict.find(m_node_name + "MEAN") != state_dict.end(), "Miss BN MEAN", m_node_name);
    ASSERT(state_dict.find(m_node_name + "VARIANCE") != state_dict.end(), "Miss BN VARIANCE", m_node_name);
    m_mean = state_dict.find(m_node_name + "MEAN")->second;
    m_variance = state_dict.find(m_node_name + "VARIANCE")->second;
    if(m_affine) {
        ASSERT(state_dict.find(m_node_name + "W") != state_dict.end(), "Miss BN W", m_node_name);
        ASSERT(state_dict.find(m_node_name + "B") != state_dict.end(), "Miss BN B", m_node_name);
        m_weights = state_dict.find(m_node_name + "W")->second;
        m_bias = state_dict.find(m_node_name + "B")->second;
    }
}

} //nn

} //licon