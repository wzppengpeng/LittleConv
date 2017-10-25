#include "licon/optim/adam.hpp"

#include "licon/utils/etensor_args.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

namespace licon
{

namespace optim
{

std::unique_ptr<Optimizer> Adam::CreateAdam(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F beta1, F beta2, F weight_decay, F eps) {
    return std::unique_ptr<Optimizer>(new Adam(std::move(register_weights), lr, beta1, beta2, weight_decay, eps));
}

void Adam::Step() {
    m_step += 1;
    wzp::ParallelForeach(std::begin(m_register_weights), std::end(m_register_weights),
        [this](std::pair<utils::ETensor<F>*, utils::ETensor<F>* >& grad_pair) {
            auto& W = *grad_pair.first;
            const auto& W_grad = *grad_pair.second;
            Update(W, W_grad);
        });
}

void Adam::Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad) {
    assert(W.count() == W_grad.count());
    utils::ETensor<F>& exq_ave = m_mt[&W];
    utils::ETensor<F>& exq_ave_sq = m_vt[&W];
    // get ptrs
    F* W_ptr = W.mutable_ptr();
    const F* W_grad_ptr = W_grad.ptr();
    F* exq_ave_ptr = exq_ave.mutable_ptr();
    F* exq_ave_sq_ptr = exq_ave_sq.mutable_ptr();
    // add weight decay
    for(size_t i = 0; i < W.count(); ++i) {
        F grad = W_grad_ptr[i];
        if(W.num() > 1) grad += W_ptr[i] * m_weight_decay;
        F mt = exq_ave_ptr[i] * m_beta1 + (1.0 - m_beta1) * grad;
        F vt = exq_ave_sq_ptr[i] * m_beta2 + (1.0 - m_beta2) * grad * grad;
        F denom = sqrt(vt) + m_eps;
        F bias_correction1 = 1.0 - pow(m_beta1, m_step);
        F bias_correction2 = 1.0 - pow(m_beta2, m_step);
        F step_size = m_lr * sqrt(bias_correction2) / bias_correction1;
        W_ptr[i] -= (step_size * (mt / denom));
        // update the new mt and vt
        exq_ave_ptr[i] = mt;
        exq_ave_sq_ptr[i] = vt;
    }
}


Adam::Adam(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
       F lr, F beta1, F beta2, F weight_decay, F eps)
    : Optimizer(std::move(register_weights)), m_beta1(beta1), m_beta2(beta2), m_weight_decay(weight_decay), m_eps(eps), m_step(0)
{
    m_lr = lr;
    for(auto& grad_pair : m_register_weights) {
        m_mt.emplace(grad_pair.first, utils::ETensor<F>(grad_pair.first->shape()));
        // fill with zeros
        utils::ETensorArgs<F>::fill(m_mt[grad_pair.first], 0);
        // squres
        m_vt.emplace(grad_pair.first, utils::ETensor<F>(grad_pair.first->shape()));
        // fill with zeros
        utils::ETensorArgs<F>::fill(m_vt[grad_pair.first], 0);
    }
}

} //optim

} //licon