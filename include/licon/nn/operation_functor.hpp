#ifndef LICON_NN_OPERATION_FUNCTOR_HPP_
#define LICON_NN_OPERATION_FUNCTOR_HPP_

/**
 * the operation functor, function like object
 * input a list of tensors, return a list of tensors
 */
#include <cassert>

#include <memory>

#include "licon/common.hpp"
#include "licon/utils/etensor.hpp"

namespace licon
{

namespace nn
{

template<typename Dtype>
class OpFunctor {

public:
    // the virtual deconstructor
    virtual ~OpFunctor() {}

    // the operator of ()
    virtual inline std::vector<utils::ETensor<Dtype> > operator() (const std::vector<utils::ETensor<Dtype>* >& bottom) {
        return Forward(bottom);
    }

    virtual inline std::vector<utils::ETensor<Dtype>* > operator() (const std::vector<utils::ETensor<Dtype>* >& bottom, int i) {
        return ForwardInplace(bottom);
    }

    virtual std::vector<utils::ETensor<Dtype> > Forward(const std::vector<utils::ETensor<Dtype>* >& bottom) = 0;
    // add the type for if no need to add new datas
    virtual std::vector<utils::ETensor<Dtype>* > ForwardInplace(const std::vector<utils::ETensor<Dtype>* >& bottom) = 0;

    virtual std::vector<utils::ETensor<Dtype> > Backward(const std::vector<utils::ETensor<Dtype>* >& top) = 0;
    // add the type for if no need to add new datas
    virtual std::vector<utils::ETensor<Dtype>* > BackwardInplace(const std::vector<utils::ETensor<Dtype>* >& top) = 0;

protected:
    std::vector<utils::ETensor<Dtype>* > m_inputs;

    virtual inline void save_for_backward(const std::vector<utils::ETensor<Dtype>* >& bottom) {
        m_inputs.clear();
        for(auto in : bottom) {
            m_inputs.emplace_back(in);
        }
    }

};

// the opfunctor ptr alias
typedef std::unique_ptr<OpFunctor<F> > FunctorPtr;

} //nn

} //licon

#endif /*LICON_NN_OPERATION_FUNCTOR_HPP_*/