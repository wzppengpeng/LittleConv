#ifndef RANGE_HPP_
#define RANGE_HPP_

#include <stdexcept>
#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>

/*
    the range is just an iterator, use c++11 for(auto i : range()) to iter
*/

namespace wzp{

template<class value_t>
class RangeImpl{
    class Iterator;
public:
    RangeImpl(value_t begin, value_t end, value_t step = 1): m_begin(begin), m_end(end), m_step(step)
    {
        if (step>0&&m_begin >= m_end)
            throw std::logic_error("end must greater than begin.");
        else if (step<0 && m_begin <= m_end)
            throw std::logic_error("end must less than begin.");

        m_step_end = (m_end - m_begin) / m_step;
        if(m_begin + m_step_end * m_step != m_end) ++m_step_end;
    }

    Iterator begin(){
        return Iterator(0, *this);
    }

    Iterator end(){
        return Iterator(m_step_end, *this);
    }

    value_t operator[](int s){
        return m_begin + s * m_step;
    }

    int size(){
        return m_step_end;
    }

private:
    value_t m_begin;
    value_t m_end;
    value_t m_step;
    int m_step_end;

private:
    class Iterator{
    public:
        // using difference_type = ptrdiff_t;
        /*constructor*/
        //the iterator point a value byu contain the reference of a RangeImpl's begin and step now
        Iterator(int start, RangeImpl& range) : m_current_step(start), m_range(range){
            m_current_value = m_range.m_begin + m_current_step * m_range.m_step;
        }

        /*the dereference*/
        value_t operator*(){ return m_current_value; }

        /*iterator self increasement*/
        const Iterator* operator++(){
            m_current_value += m_range.m_step;
            ++m_current_step;
            return this;
        }

        /*iterator judge the same*/
        bool operator==(const Iterator& other){
            return m_current_step == other.m_current_step;
        }

        /*iterator judge not the same*/
        bool operator!=(const Iterator& other){
            return m_current_step != other.m_current_step;
        }

        /*iterator self decreasmnet*/
        const Iterator* operator--(){
            m_current_value -= m_range.m_step;
            --m_current_step;
            return this;
        }

    private:
        value_t m_current_value;
        int m_current_step;
        RangeImpl& m_range;
    };

};

/*The range funcitons*/
//the begin end and step not the same type
template<typename T, typename V>
auto range(T begin, T end, V stepsize) -> RangeImpl<decltype(begin + end + stepsize)>{
    return RangeImpl<decltype(begin + end + stepsize)>(begin, end, stepsize);
}

template<typename T>
RangeImpl<T> range(T begin, T end){
    return RangeImpl<T>(begin, end, 1);
}

template<typename T>
RangeImpl<T> range(T end){
    return RangeImpl<T>(T(), end, 1);
}

/*print function like python*/
template<typename T>
void print(T&& t){
    std::cout<<t<<std::endl;
}

//the any print function
template<typename T, typename... Args>
void print(T&& t, Args&&... args){
    std::cout<<t<<" ";
    print(std::forward<Args>(args)...);
}

/**
 * print vector
 */
template<typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << '[';
    for(int i = 0; i < v.size() - 1; ++i) {
        std::cout << v[i] << ' ';
    }
    std::cout << v.back() << ']' << std::endl;
}

template<typename T>
/**
 * [print_err description]
 * @param t any type which can use <<
 */
void print_err(T&& t) {
    std::cerr<<t<<std::endl;
}

template<typename T, typename... Args>
void print_err(T&& t, Args&&... args) {
    std::cerr<<t<<" ";
    print_err(std::forward<Args>(args)...);
}

/*for now test max*/
template<typename Return=double, typename T, typename U>
Return max(T&& l, U&& r) {
    return l > r ? l : r;
}


template<typename Return=double, typename T, typename U, typename R, typename... Args>
Return max(T&& l, U&& r, R&& n, Args&&... args)
{
    auto new_l = wzp::max<Return>(std::forward<T>(l), std::forward<U>(r));
    return wzp::max<Return>(std::forward<decltype(new_l)>(new_l),
        std::forward<R>(n), std::forward<Args>(args)...);
}

/*for temp test*/
template<typename Return=double, typename T, typename U>
Return min(T&& l, U&& r) {
    return l < r ? l : r;
}


template<typename Return=double, typename T, typename U, typename R, typename... Args>
Return min(T&& l, U&& r, R&& n, Args&&... args)
{
    auto new_l = wzp::min<Return>(std::forward<T>(l), std::forward<U>(r));
    return wzp::min<Return>(std::forward<decltype(new_l)>(new_l),
        std::forward<R>(n), std::forward<Args>(args)...);
}

//the function of make_unique, just like make_shared
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/**
 * the release function, use the swap method
 */
//release the stl container's memory
template<typename T>
inline void stl_release(T& stl_container) {
    using R = typename std::remove_reference<T>::type;
    R().swap(stl_container);
}


}//wzp

#endif /*RANGE_HPP_*/