#include "licon/optim/lambda_lr.hpp"

#include <cmath>

using namespace std;

namespace licon
{

namespace optim
{

std::unique_ptr<LRScheduler> LambdaLR::CreateLambdaLR(optim::Optimizer& optimizer, int epoch_base, F lambda) {
    return std::unique_ptr<LRScheduler>(new LambdaLR(optimizer, epoch_base, lambda));
}

void LambdaLR::Step() {
    F lr = m_optimizer->get_lr();
    F lr_ = lr * (pow(m_lambda, m_epoch_now++ / m_epoch_base));
    m_optimizer->set_lr(lr_);
}

LambdaLR::LambdaLR(optim::Optimizer& optimizer, int epoch_base, F lambda)
    : LRScheduler(optimizer), m_epoch_base(epoch_base), m_lambda(lambda)
{
}




} //optim

} //licon