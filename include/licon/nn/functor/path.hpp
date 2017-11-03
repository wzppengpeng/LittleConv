#ifndef LICON_NN_PATH_FUNCTOR_HPP_
#define LICON_NN_PATH_FUNCTOR_HPP_

#include "licon/nn/operation_functor.hpp"


namespace licon
{

namespace nn
{

// the path function, get an input, then forward different copy pointers of the input
// this functor input a single bottom tensor, output the copy pointers of the botton one
// in backward, it input n grad from top, output the sum of them
class PathFunctor : public OpFunctor<F> {

public:
    // constructor
    PathFunctor(size_t path_way_num);
    //the heap constructor
    static nn::FunctorPtr CreatePathFunctor(size_t path_way_num);
    // the not implement functions
    virtual std::vector<utils::ETensor<F> > Forward(const std::vector<utils::ETensor<F>* >& bottom);
    virtual std::vector<utils::ETensor<F>* > BackwardInplace(const std::vector<utils::ETensor<F>* >& top);

    // the ancullay one
    virtual std::vector<utils::ETensor<F>* > ForwardInplace(const std::vector<utils::ETensor<F>* >& bottom);
    virtual std::vector<utils::ETensor<F> > Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    //the number of output path
    size_t m_path_way_num;
    size_t m_input_way_num = 1;

private:

    // some functions
    inline void check_forward(const std::vector<utils::ETensor<F>* >& bottom) {
        ASSERT(bottom.size() == m_input_way_num, "The path functor need only one input", bottom.size());
    }

    inline void check_backward(const std::vector<utils::ETensor<F>* >& top) {
        ASSERT(top.size() == m_path_way_num, "The path functor's grad way number mismatch", m_path_way_num);
        auto cnt = top.front()->count();
        for(auto t : top) {
            ASSERT(t->count() == cnt, "The path way grad's shape must be same", cnt, t->count());
        }
    }

};

} //nn

} //licon


#endif /*LICON_NN_PATH_FUNCTOR_HPP_*/