#ifndef LICON_OPTIM_ADAM_HPP_
#define LICON_OPTIM_ADAM_HPP_

#include <unordered_map>

#include "licon/optim/optim.hpp"

#include "licon/utils/etensor.hpp"

namespace licon
{

namespace optim
{

/**
 * the class of Adam optimizer
 */
class Adam : public Optimizer {
public:
    static std::unique_ptr<Optimizer> CreateAdam(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F beta1 = 0.9, F beta2 = 0.999, F weight_decay = 0, F eps = 0);

    void Step();


protected:
    void Update(utils::ETensor<F>& W, const utils::ETensor<F>& W_grad);
private:
    Adam(std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > register_weights,
        F lr, F beta1 = 0.9, F beta2 = 0.999, F weight_decay = 0, F eps = 0);

private:
    // the params of optimizer
    F m_beta1;
    F m_beta2;
    F m_weight_decay;
    F m_eps;
    int m_step;

    // save the previous grad
    std::unordered_map<utils::ETensor<F>*, utils::ETensor<F> > m_mt;
    std::unordered_map<utils::ETensor<F>*, utils::ETensor<F> > m_vt;
};

} //optim

} //licon


#endif /*LICON_OPTIM_ADAM_HPP_*/