#ifndef APPLY_TUPLE_HPP_
#define APPLY_TUPLE_HPP_


#include <tuple>
#include <type_traits>
#include <utility>

namespace wzp
{

namespace detail
{

template<int I, typename T, typename... Args>
struct FindIndex
{
    static int call(const std::tuple<Args...>& t, T&& val) {
        return (std::get<I - 1>(t) == val) ? (I - 1) :
            FindIndex<I - 1, T, Args...>::call(t, std::forward<T>(val));
    }
};

// the special one
template<typename T, typename... Args>
struct FindIndex<0, T, Args...>
{
    static int call(const std::tuple<Args...>& t, T&& val) {
        return (std::get<0>(t) == val) ? 0 : -1;
    }
};


/**
 * use tuple to wrap the args, then apply them to a function object
 */
//get the N - 1 arg
template<size_t N>
struct Apply
{
    template<typename F, typename T, typename... Args>
    inline static auto call(F&& f, T&& t, Args&&... a)
    -> decltype(Apply<N - 1>::call(
        std::forward<F>(f), std::forward<T>(t),
        std::get<N - 1>(std::forward<T>(t)),
        std::forward<Args>(a)...
    )) {
        return Apply<N - 1>::call(
            std::forward<F>(f), std::forward<T>(t),
            std::get<N - 1>(std::forward<T>(t)),
            std::forward<Args>(a)...
        );
    }
};

//the special
template<>
struct Apply<0>
{
    template<typename F, typename T, typename... Args>
    inline static auto call(F&& f, T&&, Args&&... a)
    ->decltype(
        std::forward<F>(f)
        (std::forward<Args>(a)...)
    ) {
        return std::forward<F>(f)
            (std::forward<Args>(a)...);
    }
};

} //detail

// the wrapper
// the wraper of function to call
template<typename T, typename... Args>
int find_index(const std::tuple<Args...>& t, T&& val) {
    return detail::FindIndex<sizeof...(Args), T, Args...>::call(t, std::forward<T>(val));
}

//the wrapper of apply
template<typename F, typename T>
inline auto apply(F&& f, T&& t)
-> decltype(
    detail::Apply<std::tuple_size<typename std::decay<T>::type>::value>
    ::call(std::forward<F>(f), std::forward<T>(t))
) {
    return detail::Apply<std::tuple_size<typename std::decay<T>::type>::value>
        ::call(std::forward<F>(f), std::forward<T>(t));
}

} //wzp


#endif /*APPLY_TUPLE_HPP_*/