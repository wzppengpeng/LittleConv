#include "licon/nn/functor/channel_concat.hpp"

#include "licon/utils/etensor_args.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

namespace licon
{

namespace nn
{

nn::FunctorPtr ChanelConcatFunctor::CreateChanelConcatFunctor() {
    return nn::FunctorPtr(new ChanelConcatFunctor());
}

std::vector<utils::ETensor<F>* > ChanelConcatFunctor::ForwardInplace(const std::vector<utils::ETensor<F>* >& bottom) {
    LOG_FATAL("this function is invalid...");
    return {};
}

std::vector<utils::ETensor<F>* > ChanelConcatFunctor::BackwardInplace(const std::vector<utils::ETensor<F>* >& top) {
    LOG_FATAL("this function is invalid...");
    return {};
}

std::vector<utils::ETensor<F> > ChanelConcatFunctor::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    utils::ETensor<F> output(m_num, std::accumulate(std::begin(m_out_indexs), std::end(m_out_indexs), 0), m_height, m_width);
    wzp::ParallelRange(m_num, [this, &output, &bottom](int i) {
        F* out_ptr = output.mutable_ptr(i);
        for(size_t j = 0; j < bottom.size(); ++j) {
            const F* input_ptr = bottom[j]->ptr(i);
            for(size_t k = 0; k < bottom[j]->count(1); ++k) {
                *out_ptr++ = input_ptr[k];
            }
        }
    });
    return {output};
}

std::vector<utils::ETensor<F> > ChanelConcatFunctor::Backward(const std::vector<utils::ETensor<F>* >& top) {
    check_backward(top);
    auto& top_grad = *top[0];
    vector<utils::ETensor<F> > bottom_grads;
    for(auto ch : m_out_indexs) {
        bottom_grads.emplace_back(utils::ETensor<F>(m_num, ch, m_height, m_width));
    }
    wzp::ParallelRange(m_num, [this, &top_grad, &bottom_grads](int i) {
        const F* grad_ptr = top_grad.ptr(i);
        for(size_t j = 0; j < m_out_indexs.size(); ++j) {
            F* bottom_ptr = bottom_grads[j].mutable_ptr(i);
            for(size_t k = 0; k < bottom_grads[j].count(1); ++k) {
                bottom_ptr[k] = *grad_ptr++;
            }
        }
    });
    return bottom_grads;
}

} //nn

} //licon