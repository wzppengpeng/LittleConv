/**
 * the single part to init the weight or bias
 */

#include "licon/utils/etensor_args.hpp"

namespace licon
{

namespace nn
{

// the static class to init the weight or bias
template<typename Dtype>
class ParamInit {

public:
    template<typename T>
    inline static void uniform(utils::ETensor<Dtype>& tensor, T a = 0, T b = 1) {
        utils::ETensorArgs<Dtype>::uniform(tensor, a, b);
    }

    // normal gassiual
    template<typename T>
    inline static void normal(utils::ETensor<Dtype>& tensor, T mean = 0, T var = 1) {
        utils::ETensorArgs<Dtype>::normal(tensor, mean, var);
    }

    // the constant init
    template<typename T>
    inline static void constant(utils::ETensor<Dtype>& tensor, T val = 0) {
        utils::ETensorArgs<Dtype>::fill(tensor, val);
    }

    // xavier_uniform
    // U (-a, a)
    inline static void xavier_uniform(utils::ETensor<Dtype>& tensor) {
        int fan_in = static_cast<int>(tensor.count()) / tensor.num();
        int fan_out = static_cast<int>(tensor.count()) / tensor.channel();
        Dtype n = (fan_in + fan_out) / Dtype(2);
        Dtype a = sqrt(Dtype(3) / n);
        utils::ETensorArgs<Dtype>::uniform(tensor, -a, a);
    }

    // xavier normal
    // N(0, std) std=2/fanin fanout
    inline static void xavier_normal(utils::ETensor<Dtype>& tensor) {
        int fan_in = static_cast<int>(tensor.count()) / tensor.num();
        int fan_out = static_cast<int>(tensor.count()) / tensor.channel();
        Dtype std = sqrt(Dtype(2) / (fan_in + fan_out));
        utils::ETensorArgs<Dtype>::normal(tensor, 0, std);
    }

    // kaiming uniform
    // U(-bound, bound)
    template<typename T>
    inline static void kaiming_uniform(utils::ETensor<Dtype>& tensor, T a = 0) {
        int fan_in = static_cast<int>(tensor.count()) / tensor.num();
        Dtype bound = sqrt(Dtype(6) / (Dtype(1 + a * a) * fan_in));
        utils::ETensorArgs<Dtype>::uniform(tensor, -bound, bound);
    }

    // kaiming noraml
    // N(0, std) std = 2 / (1 + a * a) * fanin
    template<typename T>
    inline static void kaiming_normal(utils::ETensor<Dtype>& tensor, T a = 0) {
        int fan_in = static_cast<int>(tensor.count()) / tensor.num();
        Dtype std = sqrt(Dtype(2) / (Dtype(1 + a * a) * fan_in));
        utils::ETensorArgs<Dtype>::normal(tensor, 0, std);
    }

};

} //nn

} //licon
