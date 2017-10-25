#include "licon/optim/sgd.hpp"

#include "licon/utils/etensor_args.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

namespace licon
{

namespace optim
{

std::unique_ptr<Optimizer> SGD::CreateSGD(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F momentum, F weight_decay, bool use_nesterov) {
    return std::unique_ptr<Optimizer>(new SGD(std::move(register_weights), lr, momentum, weight_decay, use_nesterov));
}

void SGD::Step() {
    // parrallel each parameters
    wzp::ParallelForeach(std::begin(m_register_weights), std::end(m_register_weights),
        [this] (std::pair<utils::ETensor<F>*, utils::ETensor<F>* >& grad_pair) {
            auto& W = *grad_pair.first;
            const auto& W_grad = *grad_pair.second;
            Update(W, W_grad);
        });
}

void SGD::Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad) {
    assert(W.count() == W_grad.count());
    utils::ETensor<F>& W_prev_grad = m_prev_grad[&W];
    F* W_ptr = W.mutable_ptr();
    const F* W_grad_ptr = W_grad.ptr();
    F* W_prev_grad_ptr = W_prev_grad.mutable_ptr();
    for(size_t i = 0; i < W.count(); ++i) {
        F d_p = W_grad_ptr[i];
        if(W.num() > 1) d_p += W_ptr[i] * m_weight_decay; //remove the bias
        F V = m_momentum * W_prev_grad_ptr[i] + d_p;
        if(m_use_nesterov) {
            d_p += (m_momentum * V);
        } else {
            d_p = V;
        }
        // step update
        W_ptr[i] -= (m_lr * d_p);
        // save the new v
        W_prev_grad_ptr[i] = V;
    }
}

SGD::SGD(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F momentum, F weight_decay, bool use_nesterov)
        : Optimizer(std::move(register_weights)), m_momentum(momentum), m_weight_decay(weight_decay), m_use_nesterov(use_nesterov)
{
    m_lr = lr;
    // create the prev grad map
    for(auto& grad_pair : m_register_weights) {
        m_prev_grad.emplace(grad_pair.first, utils::ETensor<F>(grad_pair.first->shape()));
        // fill the prev grad with zeros
        utils::ETensorArgs<F>::fill(m_prev_grad[grad_pair.first], 0.0);
    }
}

} //optim

} //licon