#ifndef LICON_E_TENSOR_ARGS_HPP_
#define LICON_E_TENSOR_ARGS_HPP_

#include "licon/utils/etensor.hpp"
#include "licon/utils/math.hpp"

#include <algorithm>

/**
 * the tensor args, to fill or some simple computations
 */

namespace licon
{

namespace utils
{

template<typename Dtype>
class ETensorArgs {

public:
    /**
     * fill the tensor with some values
     */
    // fill the tensor with a scalar value
    inline static void fill(ETensor<Dtype>& tensor, Dtype val) {
        Dtype* ptr = tensor.mutable_ptr(0);
        std::for_each(ptr, ptr + tensor.count(), [val] (Dtype& d) { d = val; });
    }

    // fill the tensor with normal distribution
    inline static void normal(ETensor<Dtype>& tensor, Dtype mean, Dtype var) {
        Dtype* ptr = tensor.mutable_ptr(0);
        std::for_each(ptr, ptr + tensor.count(),
            [mean, var] (Dtype& d) { d = licon_random_normal(mean, var); });
    }

    // fill the tensor with uniform distribution
    inline static void uniform(ETensor<Dtype>& tensor, Dtype low, Dtype high) {
        Dtype* ptr = tensor.mutable_ptr(0);
        std::for_each(ptr, ptr + tensor.count(),
            [low, high] (Dtype& d) { d = licon_random_real(low, high); });
    }

    // fill the tensor with bernoulli values
    template<typename T>
    inline static void bernoulli(ETensor<Dtype>& tensor, T p) {
        Dtype* ptr = tensor.mutable_ptr(0);
        std::for_each(ptr, ptr + tensor.count(),
            [p](Dtype& d) { d = licon_random_bernoulli(p); });
    }

    // copy the value from a ptr
    template<typename T>
    inline static void copy(ETensor<Dtype>& tensor, const T* p) {
        Dtype* ptr = tensor.mutable_ptr(0);
        for(size_t i = 0; i < tensor.count(); ++i) {
            ptr[i] = static_cast<Dtype>(p[i]);
        }
    }

    // generate a scalar
    template<typename T>
    inline static ETensor<Dtype> generate_scalar(T val) {
        ETensor<Dtype> scalar(1, 1, 1, 1);
        scalar(0, 0, 0, 0) = static_cast<Dtype>(val);
        return std::move(scalar);
    }

};

} //utils

} //licon



#endif /*LICON_E_TENSOR_ARGS_HPP_*/