#ifndef LICON_OPTIM_OPTIMIZER_HPP
#define LICON_OPTIM_OPTIMIZER_HPP


#include <vector>
#include <utility>
#include <memory>

#include "licon/common.hpp"

/**
 * the abstrct interface of optimizer
 */
namespace licon
{

namespace utils
{
template<typename Dtype>
class ETensor;
} //utils

namespace optim
{

/**
 * base class of optimizer
 */
class Optimizer {
public:
    // the creator
    Optimizer(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights)
     : m_register_weights(std::move(register_weights)) {}

    // the virtual deconstructoor
    virtual ~Optimizer() {}

    // the update interface
    virtual void Step() = 0;

    // the interface to set and get the lr
    virtual inline void set_lr(F lr) { m_lr = lr; }
    virtual inline F get_lr() const { return m_lr; }


protected:
    // the members of weights need to update, the first is the weights the second is the grad, the shape should be the same
    std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > m_register_weights;

    // the function to update each weight
    virtual void Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad) = 0;

    // learning rate
    F m_lr;

};

} //optim

} //licon


#endif /*LICON_OPTIM_OPTIMIZER_HPP*/