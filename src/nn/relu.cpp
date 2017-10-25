#include "licon/nn/node/relu.hpp"

#include "licon/utils/math.hpp"

#include "thread/parallel_algorithm.hpp"

#include "function/help_function.hpp"

using namespace std;

namespace licon
{

namespace nn
{

std::unique_ptr<OpNode<F> > Relu::CreateRelu(F negative_slope) {
    return std::unique_ptr<OpNode<F> >(new Relu(negative_slope));
}

Relu::Relu(F negative_slope) : NeuronNode<F>(), m_bottom_data(nullptr), m_negative_slope(negative_slope) {
    NeuronNode<F>::Resize();
}

void Relu::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    assert(bottom.size() == 1);
    check_forward(bottom);
    // reshape the output data
    m_data.Reshape(bottom[0]->shape());
    const F* bottom_ptr = bottom[0]->ptr(0);
    F* data_ptr = m_data.mutable_ptr(0);
    wzp::ParallelRange(m_data.count(),
        [data_ptr, bottom_ptr, this](size_t i) {
            data_ptr[i] = utils::licon_relu(bottom_ptr[i], m_negative_slope);
    });
    m_bottom_data = bottom[0];
}

void Relu::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr && top.size() == 1);
    check_backward(top);
    // reshape the output grad
    m_grad.Reshape(top[0]->shape());
    const F* bottom_ptr = m_bottom_data->ptr(0);
    const F* top_ptr = top[0]->ptr(0);
    F* grad_ptr = m_grad.mutable_ptr(0);
    wzp::ParallelRange(m_grad.count(), [grad_ptr, bottom_ptr, top_ptr, this](size_t i) {
        grad_ptr[i] = bottom_ptr[i] > 0. ? top_ptr[i] : m_negative_slope * top_ptr[i];
    });
    m_bottom_data = nullptr;
}

} //nn

} //licon