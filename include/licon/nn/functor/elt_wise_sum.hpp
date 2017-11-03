#ifndef LICON_NN_FUNCTOR_ELT_WISE_SUM_HPP_
#define LICON_NN_FUNCTOR_ELT_WISE_SUM_HPP_

#include "licon/nn/operation_functor.hpp"


namespace licon
{

namespace nn
{

// the element wise sum functor
// add all path's tensor's element together
// add backward the grad pointer for each path

class EltWiseSumFunctor : public OpFunctor<F> {

public:
    // the heap creater
    static nn::FunctorPtr CreateEltWiseSumFunctor();

    // the not implement functions
    virtual std::vector<utils::ETensor<F>* > ForwardInplace(const std::vector<utils::ETensor<F>* >& bottom);
    virtual std::vector<utils::ETensor<F> > Backward(const std::vector<utils::ETensor<F>* >& top);

    // the actual functions
    virtual std::vector<utils::ETensor<F> > Forward(const std::vector<utils::ETensor<F>* >& bottom);
    virtual std::vector<utils::ETensor<F>* > BackwardInplace(const std::vector<utils::ETensor<F>* >& top);


protected:
    size_t m_input_way_num;
    size_t m_output_way_num = 1;

private:
    // the constructor
    // EltWiseSumFunctor();

    inline void check_forward(const std::vector<utils::ETensor<F>* >& bottom) {
        m_input_way_num = bottom.size();
        size_t cnt = bottom.front()->count();
        for(auto t : bottom) {
            ASSERT(t->count() == cnt, "The input shape mismatch", cnt, t->count());
        }
    }

    inline void check_backward(const std::vector<utils::ETensor<F>* >& top) {
        ASSERT(top.size() == m_output_way_num, "The output grad way number should be one", top.size());
    }

};

} //nn

} //licon


#endif /*LICON_NN_FUNCTOR_ELT_WISE_SUM_HPP_*/