#ifndef FUNCION_TOOL_HPP_
#define FUNCION_TOOL_HPP_

#include <utility>
#include <string>

using std::declval;

namespace wzp{

/*a wise function wrapper, receive any type param*/
template<class Func, class... Args, class = typename std::enable_if<!std::is_member_function_pointer<Func>::value>::type>
inline auto function_wrapper(Func&& f, Args&&... args)
 -> decltype(f(std::forward<Args>(args)...)){
    return f(std::forward<Args>(args)...);
}


/*the wrapper for class function*/
template<class R, class C, class... DArgs, class P, class... Args>
inline auto function_wrapper(R(C::*f)(DArgs...), P && p, Args && ... args)
 -> decltype((*p.*f)(std::forward<Args>(args)...))
{
    return (*p.*f)(std::forward<Args>(args)...);
}


/*
the compose function, get only one param and a list of function,
return the list function called result
 */
template<typename OutFun, typename InnerFun>
class Composed
{
public:
    explicit Composed(OutFun out, InnerFun inner) : m_out_fn(out), m_inner_fn(inner) {}

public:
    //the operater (), for function object, this is to say that the function can only recieve
    //the arg by value, and only one value
    template<typename Arg>
    auto operator()(Arg arg)
    -> decltype(declval<OutFun>()((declval<InnerFun>()(declval<Arg>())))) {
        return m_out_fn(m_inner_fn(arg));
    }

private:
    OutFun m_out_fn;
    InnerFun m_inner_fn;
};

//the function for using
template<typename Func1, typename Func2>
Composed<Func1, Func2> compose(Func1 fun1, Func2 fun2) {
    return Composed<Func1, Func2>(fun1, fun2);
}

template<typename Func1, typename Func2, typename Func3, typename... Fs>
auto compose(Func1 f1, Func2 f2, Func3 f3, Fs... fs)
 -> decltype(compose(compose(f1, f2), f3, fs...)) {
    return compose(compose(f1, f2), f3, fs...);
}

/*
*******************
some meta functions
******************
*/
/*max function*/
template<typename T, T... list>
struct meta_max;

template<typename T, T l, T r>
struct meta_max<T, l, r>
{
    const static T value = l > r ? l : r;
};

template<typename T, T l, T r, T... rest>
struct meta_max<T, l, r, rest...>
{
    const static T temp = meta_max<T, l, r>::value;
    const static T value = meta_max<T, temp, rest...>::value;
};
/*--------------------------------*/

/*meta min function*/
template<typename T, T... list> struct meta_min;

template<typename T, T l, T r>
struct meta_min<T, l, r>
{
    const static int value = l < r ? l : r;
};

template<typename T, T l, T r, T... rest>
struct meta_min<T, l, r, rest...>
{
    const static T temp = meta_min<T, l, r>::value;
    const static T value = meta_min<T, temp, rest...>::value;
};
/*------------------------------*/

/*the pow function*/
template<size_t n, size_t m>
struct meta_pow
{
    enum { value = n * meta_pow<n, m - 1>::value};
};

template<size_t n>
struct meta_pow<n, 0>
{
    enum { value = 1 };
};

}//wzp

//use macro to easy the call process
#define MetaPow(n, m) wzp::meta_pow<n, m>::value

#endif /*FUNCION_TOOL_HPP_*/