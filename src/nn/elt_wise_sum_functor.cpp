#include "licon/nn/functor/elt_wise_sum.hpp"

#include "licon/utils/etensor_args.hpp"


using namespace std;

namespace licon
{

namespace nn
{

nn::FunctorPtr EltWiseSumFunctor::CreateEltWiseSumFunctor() {
    return FunctorPtr(new EltWiseSumFunctor());
}

std::vector<utils::ETensor<F>* > EltWiseSumFunctor::ForwardInplace(const std::vector<utils::ETensor<F>* >& bottom) {
    LOG_FATAL("this function is invalid...");
    return {};
}

std::vector<utils::ETensor<F> > EltWiseSumFunctor::Backward(const std::vector<utils::ETensor<F>* >& top) {
    LOG_FATAL("this function is invalid...");
    return {};
}

std::vector<utils::ETensor<F> > EltWiseSumFunctor::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    std::vector<utils::ETensor<F> > data_out;
    utils::ETensor<F> output(bottom.front()->shape());
    utils::ETensorArgs<F>::fill(output, 0.);
    // add all input data element wise
    for(auto b : bottom) {
        utils::ETensorArgs<F>::add(output, b->ptr());
    }
    data_out.emplace_back(std::move(output));
    return data_out;
}

std::vector<utils::ETensor<F>* > EltWiseSumFunctor::BackwardInplace(const std::vector<utils::ETensor<F>* >& top) {
    check_backward(top);
    std::vector<utils::ETensor<F>* > grad_out(m_input_way_num, top.front());
    return grad_out;
}

} //nn

} //licon