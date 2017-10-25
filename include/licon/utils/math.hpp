#ifndef LICON_MATH_HPP_
#define LICON_MATH_HPP_

#include <cmath>
#include <ctime>
#include <random>

namespace licon
{

namespace utils
{

// clip a number, let the number into the right range
template<typename T>
inline T licon_clip(const T& val, T L, T H) {
    if(val < L) {
        return L;
    } else if (val > H) {
        return H;
    } else {
        return val;
    }
}

/**
 * randome number functions
 */
// randome int(inclusive)
template<typename T = unsigned>
inline T licon_random_int(T begin, T end) {
    static std::random_device rd;
    static std::default_random_engine e(rd());
    std::uniform_int_distribution<T> u(begin, end);
    return u(e);
}

// randome real(float or double)
template<typename T = double>
inline T licon_random_real(T begin, T end) {
    static std::random_device rd;
    static std::default_random_engine er(rd());
    static std::uniform_real_distribution<T> ur(begin, end);
    return ur(er);
}

// the normal function
template<typename T>
inline T licon_random_normal(T mean, T val) {
    static std::random_device rd;
    static std::default_random_engine er(rd());
    static std::normal_distribution<T> normal(mean, val);
    return normal(er);
}

// the bernoulli ditribution
template<typename Dtype>
inline unsigned int licon_random_bernoulli(Dtype p) {
    static std::random_device rd;
    static std::default_random_engine er(rd());
    static std::bernoulli_distribution ber_dis(p);
    return ber_dis(er) ? 1 : 0;
}

// select a radom one except the previous one
template<typename T, typename I>
inline T licon_select_rand(T i, I begin, I end) {
    auto j = i;
    while(j == i) {
        j = licon_random_int<T>(begin, end); // includesive
    }
    return j;
}



/**
 * Math functions
 */
// down is the down number of log function
template<typename Type>
inline Type licon_log(Type val, int down) {
    return log(val) / log(down);
}

// sigmoid
template<typename Type>
inline Type licon_sigmoid(Type val) {
    return 1.0 / (1.0 + exp(-val));
}

// relu function
template<typename Type>
inline Type licon_relu(Type val, Type negative_slope = 0.) {
    return val > 0. ? val : negative_slope * val;
}

// tanh function
template<typename Type>
inline Type licon_tanh(Type val) {
    return (exp(val) - exp(-val)) / (exp(val) + exp(-val));
}

} //utils

} //licon


#endif /*LICON_MATH_HPP_*/