#include "licon/optim/rmsprop.hpp"

#include "licon/utils/etensor_args.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

namespace licon
{

namespace optim
{

std::unique_ptr<Optimizer> RMSprop::CreateRMSprop(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F alpha, F eps, F weight_decay, F momentum, bool centered) {
    return std::unique_ptr<Optimizer>(new RMSprop(std::move(register_weights), lr, alpha, eps, weight_decay, momentum, centered));
}

void RMSprop::Step() {
    m_step += 1;
    wzp::ParallelForeach(std::begin(m_register_weights), std::end(m_register_weights),
        [this](std::pair<utils::ETensor<F>*, utils::ETensor<F>* >& grad_pair) {
            auto& W = *grad_pair.first;
            const auto& W_grad = *grad_pair.second;
            Update(W, W_grad);
        });
}

void RMSprop::Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad) {
    assert(W.count() == W_grad.count());
    utils::ETensor<F>& squre_avg = m_squre_avg[&W];
    utils::ETensor<F>& buf = m_momentum_buffer[&W];
    utils::ETensor<F>& grad_avg = m_grad_avg[&W];
    // get ptrs
    F* W_ptr = W.mutable_ptr();
    const F* W_grad_ptr = W_grad.ptr();
    F* squre_avg_ptr = squre_avg.mutable_ptr();
    F* buf_ptr = buf.mutable_ptr();
    F* grad_avg_ptr = grad_avg.mutable_ptr();
    for(size_t i = 0; i < W.count(); ++i) {
        F grad = W_grad_ptr[i];
        if(W.num() > 1) grad += W_ptr[i] * m_weight_decay;
        squre_avg_ptr[i] = squre_avg_ptr[i] * m_alpha + (1. - m_alpha) * grad * grad;
        F avg = sqrt(squre_avg_ptr[i]) + m_eps;
        if(m_centered) {
            grad_avg_ptr[i] = grad_avg_ptr[i] * m_alpha + (1. - m_alpha) * grad;
            avg = sqrt(squre_avg_ptr[i] - grad_avg_ptr[i] * grad_avg_ptr[i]) + m_eps;
        }
        if(m_momentum > 0.) {
            buf_ptr[i] = buf_ptr[i] * m_momentum + (grad / avg);
            W_ptr[i] -= m_lr * buf_ptr[i];
        } else {
            W_ptr[i] -= m_lr * (grad / avg);
        }
    }
}

RMSprop::RMSprop(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F alpha, F eps, F weight_decay, F momentum, bool centered)
        : Optimizer(std::move(register_weights)), m_alpha(alpha), m_eps(eps), m_weight_decay(weight_decay), m_momentum(momentum), m_centered(centered), m_step(0)
{
    m_lr = lr;
    for(auto& grad_pair : m_register_weights) {
        m_squre_avg.emplace(grad_pair.first, utils::ETensor<F>(grad_pair.first->shape()));
        utils::ETensorArgs<F>::fill(m_squre_avg[grad_pair.first], 0.);
        m_momentum_buffer.emplace(grad_pair.first, utils::ETensor<F>(grad_pair.first->shape()));
        utils::ETensorArgs<F>::fill(m_momentum_buffer[grad_pair.first], 0.);
        m_grad_avg.emplace(grad_pair.first, utils::ETensor<F>(grad_pair.first->shape()));
        utils::ETensorArgs<F>::fill(m_grad_avg[grad_pair.first], 0.);
    }
}

} //optim

} //licon