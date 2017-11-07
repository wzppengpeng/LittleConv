#ifndef LICON_NN_FUNCTOR_CHANEL_CONCAT_HPP_
#define LICON_NN_FUNCTOR_CHANEL_CONCAT_HPP_

#include "licon/nn/operation_functor.hpp"

namespace licon
{

namespace nn
{

// the channel concat functor
// concat the feature map into one feature map
// the input fmap should be in the same size
// backward grad to the related parts
class ChanelConcatFunctor : public OpFunctor<F> {

public:
    // the heap creator
    static nn::FunctorPtr CreateChanelConcatFunctor();

    // the not implement functions
    virtual std::vector<utils::ETensor<F>* > ForwardInplace(const std::vector<utils::ETensor<F>* >& bottom);
    virtual std::vector<utils::ETensor<F>* > BackwardInplace(const std::vector<utils::ETensor<F>* >& top);


    // the actual functions
    virtual std::vector<utils::ETensor<F> > Forward(const std::vector<utils::ETensor<F>* >& bottom);
    virtual std::vector<utils::ETensor<F> > Backward(const std::vector<utils::ETensor<F>* >& top);


protected:
    // the input num and output num
    size_t m_input_way_num;
    size_t m_output_way_num = 1;
    // the index to record the output data's index
    std::vector<int> m_out_indexs;

private:
    int m_num;
    int m_height;
    int m_width;

    // the check functions
    inline void check_forward(const std::vector<utils::ETensor<F>* >& bottom) {
        m_input_way_num = bottom.size();
        m_num = bottom.front()->num();
        m_height = bottom.front()->height();
        m_width = bottom.front()->width();
        m_out_indexs.clear();
        for(auto t : bottom) {
            ASSERT(t->num() == m_num, "The Input Num mismatch", m_num, t->num());
            ASSERT(t->height() == m_height, "The Input Height Mismatch", m_height, t->height());
            ASSERT(t->width() == m_width, "The Input Width Mismatch", m_width, t->width());
            m_out_indexs.emplace_back(t->channel());
        }
    }


    inline void check_backward(const std::vector<utils::ETensor<F>* >& top) {
        ASSERT(top.size() == m_output_way_num, "The output grad way number should be one", top.size());
        ASSERT(top.front()->num() == m_num, "The Top Grad Number Mismatch", m_num);
        ASSERT(top.front()->height() == m_height, "The Top Grad Number Mismatch", m_height);
        ASSERT(top.front()->width() == m_width, "The Top Grad Number Mismatch", m_width);
    }


};

} //nn

} //licon


#endif /*LICON_NN_FUNCTOR_CHANEL_CONCAT_HPP_*/