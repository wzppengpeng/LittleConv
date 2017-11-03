#include "licon/nn/functor/path.hpp"

#include "licon/utils/etensor_args.hpp"

using namespace std;

namespace licon
{

namespace nn
{

nn::FunctorPtr PathFunctor::CreatePathFunctor(size_t path_way_num) {
    return FunctorPtr(new PathFunctor(path_way_num));
}

std::vector<utils::ETensor<F> > PathFunctor::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    LOG_FATAL("this function is invalid...");
    return {};
}

std::vector<utils::ETensor<F>* > PathFunctor::BackwardInplace(const std::vector<utils::ETensor<F>* >& top) {
    LOG_FATAL("this function is invalid...");
    return {};
}

std::vector<utils::ETensor<F>* > PathFunctor::ForwardInplace(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    std::vector<utils::ETensor<F>* > output(m_path_way_num, bottom[0]);
    return output;
}

std::vector<utils::ETensor<F> > PathFunctor::Backward(const std::vector<utils::ETensor<F>* >& top) {
    check_backward(top);
    std::vector<utils::ETensor<F> > grad_out;
    utils::ETensor<F> out(top.front()->shape());
    utils::ETensorArgs<F>::fill(out, 0);
    for(auto grad_in : top) {
        utils::ETensorArgs<F>::add(out, grad_in->ptr());
    }
    grad_out.emplace_back(std::move(out));
    return grad_out;
}

PathFunctor::PathFunctor(size_t path_way_num) : m_path_way_num(path_way_num) {}

} //nn

} //licon