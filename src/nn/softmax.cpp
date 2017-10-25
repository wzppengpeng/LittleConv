#include "licon/nn/node/softmax.hpp"

#include "licon/utils/math.hpp"

// #include "container/ematrix.hpp"

#include "thread/parallel_algorithm.hpp"

#include "function/help_function.hpp"

using namespace std;

namespace licon
{

namespace nn
{

std::unique_ptr<OpNode<F> > Softmax::CreateSoftmax() {
    return std::unique_ptr<OpNode<F> >(new Softmax());
}

void Softmax::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    ASSERT(bottom[0]->height() == 1 && bottom[0]->width() == 1, "the softmax node should be behind the linear fc");
    m_bottom_data = bottom[0];
    // reshape the data output
    m_data.Reshape(m_bottom_data->shape());
    wzp::ParallelRange(m_bottom_data->num(), [this](int i) {
        // first to find the max val in the i row
        auto max_activation = *(std::max_element(m_bottom_data->ptr(i),
            m_bottom_data->ptr(i) + m_bottom_data->channel()));
        F sum_val = 0.0;
        for(int j = 0; j < m_bottom_data->channel(); ++j) {
            sum_val += exp(m_bottom_data->at(i, j, 0, 0) - max_activation);
        }
        for(int j = 0; j < m_bottom_data->channel(); ++j) {
            m_data(i, j, 0, 0) = exp(m_bottom_data->at(i, j, 0, 0) - max_activation) / sum_val;
        }
    });
}

void Softmax::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ASSERT(top[0]->height() == 1 && top[0]->width() == 1, "the softmax node should be behind the linear fc");
    ASSERT(top[0]->count() == m_bottom_data->count(), "the shape input output miss match");
    // reshape the grad output
    m_grad.Reshape(m_bottom_data->shape());
    auto& top_grad = *(top[0]);
    wzp::ParallelRange(m_bottom_data->num(), [this, &top_grad](int i) {
        F sum_val = 0.0;
        for(int j = 0; j < m_bottom_data->channel(); ++j) {
            sum_val += m_data(i, j, 0, 0) * top_grad(i, j, 0, 0);
        }
        for(int j = 0; j < m_bottom_data->channel(); ++j) {
            m_grad(i, j, 0, 0) = top_grad(i, j, 0, 0) * m_data(i, j, 0, 0) - sum_val * m_data(i, j, 0, 0);
        }
    });
    m_bottom_data = nullptr;
}

Softmax::Softmax() : NeuronNode<F>(), m_bottom_data(nullptr) {
    NeuronNode<F>::Resize();
}

} //nn

} //licon