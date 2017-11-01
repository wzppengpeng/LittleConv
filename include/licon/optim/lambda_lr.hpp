#ifndef LICON_OPTIM_LAMBDA_LR_HPP_
#define LICON_OPTIM_LAMBDA_LR_HPP_

#include "licon/optim/lr_scheduler.hpp"


/**
 * the lambda lr schedule for like this
 *  lambda1 = lambda epoch: epoch // 30
 *  lambda2 = lambda epoch: 0.95 ** epoch
 */


namespace licon
{

namespace optim
{

class LambdaLR : public LRScheduler {

public:
    // the heap create function
    static std::unique_ptr<LRScheduler> CreateLambdaLR(optim::Optimizer& optimizer, int epoch_base, F lambda);

    virtual void Step();

private:
    LambdaLR(optim::Optimizer& optimizer, int epoch_base, F lambda);

    int m_epoch_base;
    F m_lambda;

};

} //optim

} //licon

#endif /*LICON_OPTIM_LAMBDA_LR_HPP_*/