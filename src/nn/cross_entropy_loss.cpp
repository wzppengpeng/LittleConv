#include "licon/nn/node/cross_entropy_loss.hpp"

#include <cfloat>

#include "container/ematrix.hpp"

#include "licon/utils/etensor_args.hpp"
#include "licon/utils/math.hpp"

#include "function/help_function.hpp"

using namespace std;

const static F VALUE_LOW = FLT_MIN;
const static F VALUE_HIGH = 1e100;

using wzp::print;

namespace licon
{

namespace nn
{

std::unique_ptr<OpNode<F> > CrossEntropyLoss::CreateCrossEntropyLoss(bool size_average) {
    return std::unique_ptr<OpNode<F> >(new CrossEntropyLoss(size_average));
}

void CrossEntropyLoss::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    ASSERT(static_cast<size_t>(bottom[0]->num()) == bottom[1]->count() && bottom[0]->height() == 1 && bottom[0]->width() == 1, "the data number and label number miss match"); //the input is n*c
    m_bottom_data = bottom[0];
    m_bottom_label = bottom[1];
    // reshape the data output
    // the output is scalar
    m_data.Reshape(1, 1, 1, 1);
    F loss = 0.;
    for(int i = 0; i < m_bottom_data->num(); ++i) {
        assert(static_cast<int>(m_bottom_label->at(i, 0, 0, 0)) >= 0 && static_cast<int>(m_bottom_label->at(i, 0, 0, 0)) < m_bottom_data->channel());
        loss -= log(utils::licon_clip(m_bottom_data->at(i, static_cast<int>(m_bottom_label->at(i, 0, 0, 0)), 0, 0), VALUE_LOW, VALUE_HIGH));
    }
    if(m_size_average) loss /= static_cast<F>(m_bottom_data->num());
    if(loss != loss) {
        wzp::log::fatal("The Loss Is Nan", loss);
    }
    m_data(0, 0, 0, 0) = loss;
}

void CrossEntropyLoss::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr && m_bottom_label != nullptr);
    check_backward(top);
    // check the top grad is a scalar
    ASSERT(top[0]->count() == 1, "the loss layer grad should be scalar", top[0]->count());
    auto& top_grad = *top[0];
    // reshpae the bottom grad
    m_grad.Reshape(m_bottom_data->shape());

    utils::ETensorArgs<F>::fill(m_grad, 0);
    for(int i = 0; i < m_bottom_data->num(); ++i) {
        int k = static_cast<int>(m_bottom_label->at(i, 0, 0, 0));
        m_grad(i, k, 0, 0) = -top_grad(0, 0, 0, 0) / (utils::licon_clip(m_bottom_data->at(i, k, 0, 0), VALUE_LOW, VALUE_HIGH));
        if(m_size_average) m_grad(i, k, 0, 0) /= static_cast<F>(m_bottom_data->num());
    }
    m_bottom_label = nullptr;
    m_bottom_data = nullptr;
}

CrossEntropyLoss::CrossEntropyLoss(bool size_average)
    : LossNode<F>(), m_size_average(size_average), m_bottom_data(nullptr), m_bottom_label(nullptr)
{
    // resize the bottom input
    OpNode<F>::m_former_nodes.resize(2, nullptr);
}

} //nn

} //licon