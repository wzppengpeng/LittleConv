#include "licon/nn/node/dropout.hpp"

#include "licon/utils/etensor_args.hpp"

#include "thread/parallel_algorithm.hpp"
#include "function/help_function.hpp"

using namespace std;

namespace licon
{

namespace nn
{

std::unique_ptr<OpNode<F> > Dropout::CreateDropout(F p) {
    return std::unique_ptr<OpNode<F> >(new Dropout(p));
}

void Dropout::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    m_bottom_data = bottom[0];
    if(this->m_phase == Phase::TRAIN) {
        // reshape
        m_data.Reshape(m_bottom_data->shape());
        m_mask.Reshape(m_bottom_data->shape());
        utils::ETensorArgs<unsigned int>::bernoulli(m_mask, 1. - m_threshold);
        const F* bottom_data_ptr = m_bottom_data->ptr();
        F* data_ptr = m_data.mutable_ptr();
        const unsigned int* mask_ptr = m_mask.ptr();
        wzp::ParallelRange(m_bottom_data->count(), [bottom_data_ptr, data_ptr, mask_ptr, this](size_t i) {
            data_ptr[i] = bottom_data_ptr[i] * mask_ptr[i] * m_scale;
        });
    } else {
        // copy the input to ouput
        m_data = *m_bottom_data;
    }
}

void Dropout::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr && top[0]->count() == m_bottom_data->count());
    check_backward(top);
    if(this->m_phase == Phase::TRAIN) {
        // reshape the grad
        m_grad.Reshape(m_bottom_data->shape());
        const unsigned int* mask_ptr = m_mask.ptr();
        const F* top_grad_ptr = top[0]->ptr();
        F* grad_ptr = m_grad.mutable_ptr();
        wzp::ParallelRange(m_grad.count(), [mask_ptr, top_grad_ptr, grad_ptr, this](size_t i) {
            grad_ptr[i] = top_grad_ptr[i] * mask_ptr[i] * m_scale;
        });
    } else {
        // copy the grad
        m_grad = *(top[0]);
    }
    m_bottom_data = nullptr;
}

Dropout::Dropout(F p) : NeuronNode<F>(), m_threshold(p), m_bottom_data(nullptr) {
    ASSERT(m_threshold > 0. && m_threshold < 1., "the dropout threshold not legall", m_threshold);
    NeuronNode<F>::Resize();
    m_scale = 1.0 / (1.0 - m_threshold);
 }

} //nn

} //licon