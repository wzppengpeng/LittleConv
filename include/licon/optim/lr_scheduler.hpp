#ifndef LICON_OPTIM_LR_SCHEDULER_HPP_
#define LICON_OPTIM_LR_SCHEDULER_HPP_

/**
 * provides several methods to adjust the learning rate based on the number of epoches
 */

#include <memory>

#include "licon/optim/optim.hpp"

namespace licon
{

namespace optim
{

// the virtual class for scheduler
class LRScheduler {

public:
    // the creater
    LRScheduler(optim::Optimizer& optimizer)
        : m_optimizer(&optimizer), m_epoch_now(0)
    {}

    // the virtual deconstructor
    virtual ~LRScheduler() {}

    // the update function
    virtual void Step() = 0;

    // the interface to set and get the epoch now
    virtual inline void set_epoch(int epoch) { m_epoch_now = epoch; }
    virtual inline int get_epoch() const { return m_epoch_now; }

protected:

    // the optimizer pointer
    optim::Optimizer* m_optimizer;

    // the epoch num now
    int m_epoch_now;

};

} //optim

} //licon


#endif /*LICON_OPTIM_LR_SCHEDULER_HPP_*/