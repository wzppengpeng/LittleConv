#include "licon/nn/node/selu.hpp"

#include "licon/utils/math.hpp"

#include "thread/parallel_algorithm.hpp"

/**
 * the default alpha and scale parameters
 * from "https://arxiv.org/abs/1706.02515"
 */
const static F SCALE = 1.0507009873554804934193349852946;
const static F ALPHA = 1.6732632423543772848170429916717;


namespace licon
{

namespace nn
{

nn::NodePtr SELU::CreateSELU() {
    return nn::NodePtr(new SELU);
}


void SELU::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    m_bottom_data = bottom[0];
    // reshape the data output
    m_data.Reshape(m_bottom_data->shape());
    const F* bottom_ptr = m_bottom_data->ptr();
    F* data_ptr = m_data.mutable_ptr();
    wzp::ParallelRange(m_bottom_data->count(), [this, bottom_ptr, data_ptr](size_t i) {
        data_ptr[i] = bottom_ptr[i] >= 0. ? m_scale * bottom_ptr[i] : m_scale * m_alpha * (exp(bottom_ptr[i] - 1));
    });
}


void SELU::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ASSERT(top[0]->count() == m_bottom_data->count(), "SELU shape mismatch", top[0]->count(), m_bottom_data->count());
    // reshape the grad
    m_grad.Reshape(top[0]->shape());
    const F* bottom_ptr = m_bottom_data->ptr();
    const F* top_ptr = top[0]->ptr();
    const F* data_ptr = m_data.ptr();
    F* grad_ptr = m_grad.mutable_ptr();
    wzp::ParallelRange(m_bottom_data->count(), [this, bottom_ptr, top_ptr, grad_ptr, data_ptr](size_t i) {
        grad_ptr[i] = bottom_ptr[i] >= 0. ? m_scale * top_ptr[i] : (data_ptr[i] + m_scale * m_alpha) * top_ptr[i];
    });
    m_bottom_data = nullptr;
}

// constructor
SELU::SELU() : NeuronNode<F>(), m_bottom_data(nullptr), m_scale(SCALE), m_alpha(ALPHA) {}


} //nn

} //licon