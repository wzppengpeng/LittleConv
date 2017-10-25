#ifndef LICON_OPTIM_SGD_HPP_
#define LICON_OPTIM_SGD_HPP_

#include <unordered_map>

#include "licon/optim/optim.hpp"

#include "licon/utils/etensor.hpp"

namespace licon
{

namespace optim
{

/**
 * the class of SGD optimizer
 */
class SGD : public Optimizer {

public:
    // the creator of SGD optimizer
    static std::unique_ptr<Optimizer> CreateSGD(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F momentum = 0.9, F weight_decay = 0, bool use_nesterov = false);

    void Step();

protected:
    void Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad);

private:
    SGD(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F momentum = 0.9, F weight_decay = 0, bool use_nesterov = false);

private:
    // the params of optimizer

    F m_momentum;
    F m_weight_decay;
    bool m_use_nesterov;

    // to save the previous grad
    std::unordered_map<utils::ETensor<F>*, utils::ETensor<F> > m_prev_grad;

};

} //optim

} //licon

#endif /*LICON_OPTIM_SGD_HPP_*/