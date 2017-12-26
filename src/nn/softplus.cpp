#include "licon/nn/node/softplus.hpp"

#include "licon/utils/math.hpp"

#include "thread/parallel_algorithm.hpp"


using namespace std;

namespace licon
{

namespace nn
{

nn::NodePtr Softplus::CreateSoftplus(F beta, F threshold) {
    return nn::NodePtr(new Softplus(beta, threshold));
}

void Softplus::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    m_bottom_data = bottom[0];
    // reshape the output data
    m_data.Reshape(m_bottom_data->shape());
    const F* bottom_ptr = m_bottom_data->ptr();
    F* data_ptr = m_data.mutable_ptr();
    wzp::ParallelRange(m_bottom_data->count(), [this, bottom_ptr, data_ptr](size_t i) {
        data_ptr[i] = bottom_ptr[i] >= m_threshold ? bottom_ptr[i] : (1. / m_beta) * log(1. + exp(m_beta * bottom_ptr[i]));
    });
}

void Softplus::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ASSERT(top[0]->count() == m_bottom_data->count(), "Softplus shape mismatch", top[0]->count(), m_bottom_data->count());
    // reshape the grad
    m_grad.Reshape(m_bottom_data->shape());
    const F* bottom_ptr = m_bottom_data->ptr();
    const F* top_ptr = top[0]->ptr();
    F* grad_ptr = m_grad.mutable_ptr();
    wzp::ParallelRange(m_bottom_data->count(), [this, bottom_ptr, top_ptr, grad_ptr](size_t i) {
        grad_ptr[i] = bottom_ptr[i] >= m_threshold ? top_ptr[i]
            : top_ptr[i] * ((1. / m_beta) * (exp(m_beta * bottom_ptr[i])) / (1. + exp(m_beta * bottom_ptr[i])));
    });
    m_bottom_data = nullptr;
}

Softplus::Softplus(F beta, F threshold) : NeuronNode<F>(), m_beta(beta), m_threshold(threshold) {}

} //nn

} //licon