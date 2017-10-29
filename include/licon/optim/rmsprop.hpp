#ifndef LICON_OPTIM_RMSPROP_HPP_
#define LICON_OPTIM_RMSPROP_HPP_

#include <unordered_map>

#include "licon/optim/optim.hpp"

#include "licon/utils/etensor.hpp"

namespace licon
{

namespace optim
{

/**
 * the optimizer class of rmsprop
 */
class RMSprop : public Optimizer {

public:
    // the heap object creater
    static std::unique_ptr<Optimizer> CreateRMSprop(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F alpha=0.99, F eps=1e-8, F weight_decay=0., F momentum=0., bool centered=false);

    void Step();

protected:
    void Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad);

private:
    RMSprop(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F alpha=0.99, F eps=1e-8, F weight_decay=0., F momentum=0., bool centered=false);

    // the params  of optimizer
    F m_alpha;
    F m_eps;
    F m_weight_decay;
    F m_momentum;
    bool m_centered;

    int m_step;

    // save the previous grad
    std::unordered_map<utils::ETensor<F>*, utils::ETensor<F> > m_squre_avg;
    std::unordered_map<utils::ETensor<F>*, utils::ETensor<F> > m_momentum_buffer;
    std::unordered_map<utils::ETensor<F>*, utils::ETensor<F> > m_grad_avg;

};

} //optim

} //licon

#endif /*LICON_OPTIM_RMSPROP_HPP_*/