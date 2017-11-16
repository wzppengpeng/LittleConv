#include "licon/nn/node/elu.hpp"

#include "licon/utils/math.hpp"

#include "thread/parallel_algorithm.hpp"


namespace licon
{

namespace nn
{

nn::NodePtr ELU::CreateELU(F alpha) {
    return nn::NodePtr(new ELU(alpha));
}

void ELU::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    m_bottom_data = bottom[0];
    // reshape the output
    m_data.Reshape(m_bottom_data->shape());
    const F* bottom_ptr = m_bottom_data->ptr();
    F* data_ptr = m_data.mutable_ptr();
    wzp::ParallelRange(m_bottom_data->count(), [this, bottom_ptr, data_ptr](size_t i) {
        data_ptr[i] = bottom_ptr[i] >= 0. ? bottom_ptr[i] : (m_alpha * (exp(bottom_ptr[i]) - 1));
    });
}

void ELU::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ASSERT(top[0]->count() == m_bottom_data->count(), "ELU shape mismatch", top[0]->count(), m_bottom_data->count());
    // reshape the grad
    m_grad.Reshape(top[0]->shape());
    const F* bottom_ptr = m_bottom_data->ptr();
    const F* top_ptr = top[0]->ptr();
    const F* data_ptr = m_data.ptr();
    F* grad_ptr = m_grad.mutable_ptr();
    wzp::ParallelRange(m_bottom_data->count(), [this, bottom_ptr, top_ptr, grad_ptr, data_ptr](size_t i) {
        grad_ptr[i] = bottom_ptr[i] >= 0. ? top_ptr[i] : (data_ptr[i] + m_alpha) * top_ptr[i];
    });
    m_bottom_data = nullptr;
}

ELU::ELU(F alpha) : NeuronNode<F>(), m_bottom_data(nullptr), m_alpha(alpha) {}

} //nn

} //licon