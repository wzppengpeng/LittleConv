#include "licon/nn/node/tanh.hpp"

#include "licon/utils/math.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

namespace licon
{

namespace nn
{

std::unique_ptr<OpNode<F> > Tanh::CreateTanh() {
    return std::unique_ptr<OpNode<F> >(new Tanh());
}

void Tanh::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    // resize the output data
    m_data.Reshape(bottom[0]->shape());
    const F* bottom_ptr = bottom[0]->ptr(0);
    F* data_ptr = m_data.mutable_ptr(0);
    wzp::ParallelRange(m_data.count(), [data_ptr, bottom_ptr](size_t i) {
        data_ptr[i] = utils::licon_tanh(bottom_ptr[i]);
    });
    m_has_forwarded = true;
}

void Tanh::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_has_forwarded);
    check_backward(top);
    // reshape the grad
    m_grad.Reshape(top[0]->shape());
    const F* top_ptr = top[0]->ptr();
    F* grad_ptr = m_grad.mutable_ptr();
    const F* data_ptr = m_data.ptr();
    wzp::ParallelRange(m_grad.count(), [grad_ptr, data_ptr, top_ptr](size_t i) {
        grad_ptr[i] = top_ptr[i] * (1.0 - (data_ptr[i] * data_ptr[i]));
    });
    m_has_forwarded = false;
}

Tanh::Tanh() : NeuronNode<F>(), m_has_forwarded(false) {
    NeuronNode<F>::Resize();
}

} //nn

} //licon