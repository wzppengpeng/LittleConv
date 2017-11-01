#ifndef LICON_OPTIM_STEP_HPP_
#define LICON_OPTIM_STEP_HPP_


#include "licon/optim/lr_scheduler.hpp"


/**
 * lr = lr * gamma
 */

namespace licon
{

namespace optim
{

class StepLR : public LRScheduler {

public:
    // the heap create function
    static std::unique_ptr<LRScheduler> CreateStepLR(optim::Optimizer& optimizer, int step_size, F gamma);

    virtual void Step();

private:
    StepLR(optim::Optimizer& optimizer, int step_size, F gamma);

    int m_step_size;
    F m_gamma;

};

} //optim

} //licon

#endif /*LICON_OPTIM_STEP_HPP_*/