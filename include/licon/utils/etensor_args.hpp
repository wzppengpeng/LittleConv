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

namespace details
{

// vector div, a vector div a scalar
template<typename Dtype>
inline void vector_div(std::vector<Dtype>& x, Dtype denom) {
    std::for_each(std::begin(x), std::end(x), [denom](Dtype& val) { val /= denom; });
}

} //details

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

    // add the value from a ptr
    template<typename T>
    inline static void add(ETensor<Dtype>& tensor, const T* p) {
        Dtype* ptr = tensor.mutable_ptr();
        for(size_t i = 0; i < tensor.count(); ++i) {
            ptr[i] += static_cast<Dtype>(p[i]);
        }
    }

    // generate a scalar
    template<typename T>
    inline static ETensor<Dtype> generate_scalar(T val) {
        ETensor<Dtype> scalar(1, 1, 1, 1);
        scalar(0, 0, 0, 0) = static_cast<Dtype>(val);
        return std::move(scalar);
    }

    // calculate the channel sum of a 4-D tensor
    inline static void channel_sum(const ETensor<Dtype>& tensor, Dtype* sum_ptr) {
        int spatial_dim = tensor.count(2);
        for(int i = 0; i < tensor.num(); ++i) {
            for(int j = 0; j < tensor.channel(); ++j) {
                auto& rmean = sum_ptr[j];
                const auto* it = tensor.ptr(i) + (j * spatial_dim);
                rmean = std::accumulate(it, it + spatial_dim, rmean);
            }
        }
    }

    // calculate the channel sum of a 4-d tensor
    inline static void channel_sum(const ETensor<Dtype>& tensor, std::vector<Dtype>& sum_vec) {
        sum_vec.clear();
        sum_vec.resize(tensor.channel(), 0);
        channel_sum(tensor, &sum_vec[0]);
    }

    // calculate the channel mean of a 4-d tensor
    inline static void channel_mean(const ETensor<Dtype>& tensor, Dtype* mean_ptr) {
        int spatial_dim = tensor.count(2);
        channel_sum(tensor, mean_ptr);
        for(int i = 0; i < tensor.channel(); ++i) {
            mean_ptr[i] /= static_cast<Dtype>(spatial_dim * tensor.num());
        }
    }

    // calculate the channel mean of a 4-d tensor
    inline static void channel_mean(const ETensor<Dtype>& tensor, std::vector<Dtype>& mean_vec) {
        int spatial_dim = tensor.count(2);
        channel_sum(tensor, mean_vec);
        details::vector_div(mean_vec, static_cast<Dtype>(spatial_dim * tensor.num()));
    }

    // calculate the channel mean and variance of a 4-d tensor
    inline static void channel_variance(const ETensor<Dtype>& tensor, Dtype* mean_ptr, Dtype* variance_ptr) {
        channel_mean(tensor, mean_ptr);
        int spatial_dim = tensor.count(2);
        for(int i = 0; i < tensor.num(); ++i) {
            for(int j = 0; j < tensor.channel(); ++j) {
                auto& rvar = variance_ptr[j];
                const auto* it = tensor.ptr(i) + (j * spatial_dim);
                const auto ex = mean_ptr[j];
                rvar = std::accumulate(it, it + spatial_dim, rvar,
                    [ex](Dtype current, Dtype x) { return current + pow(x - ex, 2.); });
            }
        }
        for(int i = 0; i < tensor.channel(); ++i) {
            variance_ptr[i] /= static_cast<Dtype>(spatial_dim * tensor.num());
        }
    }

    // calculate the channel mean and variance of a 4-d tensor
    inline static void channel_variance(const ETensor<Dtype>& tensor, std::vector<Dtype>& mean_vec, std::vector<Dtype>& variance_vec) {
        channel_mean(tensor, mean_vec);
        variance_vec.clear();
        variance_vec.resize(tensor.channel(), Dtype(0));
        int spatial_dim = tensor.count(2);
        for(int i = 0; i < tensor.num(); ++i) {
            for(int j = 0; j < tensor.channel(); ++j) {
                auto& rvar = variance_vec[j];
                const auto* it = tensor.ptr(i) + (j * spatial_dim);
                const auto ex = mean_vec[j];
                rvar = std::accumulate(it, it + spatial_dim, rvar,
                    [ex](Dtype current, Dtype x) { return current + pow(x - ex, 2.); });
            }
        }
        details::vector_div(variance_vec, std::max(Dtype(1), static_cast<Dtype>(tensor.num() * spatial_dim - Dtype(1))));
    }


};

} //utils

} //licon



#endif /*LICON_E_TENSOR_ARGS_HPP_*/