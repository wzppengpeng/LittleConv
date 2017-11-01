#include "licon/optim/step_lr.hpp"


using namespace std;

namespace licon
{

namespace optim
{

std::unique_ptr<LRScheduler> StepLR::CreateStepLR(optim::Optimizer& optimizer, int step_size, F gamma) {
    return std::unique_ptr<LRScheduler>(new StepLR(optimizer, step_size, gamma));
}

void StepLR::Step() {
    if(m_epoch_now > 0 && m_epoch_now % m_step_size == 0) {
        F lr = m_optimizer->get_lr();
        F lr_ = m_gamma * lr;
        m_optimizer->set_lr(lr_);
    }
    ++m_epoch_now;
}

StepLR::StepLR(optim::Optimizer& optimizer, int step_size, F gamma)
    : LRScheduler(optimizer), m_step_size(step_size), m_gamma(gamma)
{
}

} //optim

} //licon