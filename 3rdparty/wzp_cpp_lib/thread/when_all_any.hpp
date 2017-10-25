#ifndef LIB_WHEN_ALL_ANY_HPP
#define LIB_WHEN_ALL_ANY_HPP

#include <vector>
#include <chrono>

#include "thread/task.hpp"


using std::vector;
using std::pair;

namespace wzp {

//the when_all function,
//input a range stl type(vector<task<R()>>)
//output the a Task<vector()> ==> just a task, function object, none block
//use get or run get the answer=>a vector save the same type return answer
template<typename Range>
Task<std::vector<typename Range::value_type::return_type>()> WhenAll(Range& range) {
    typedef typename Range::value_type::return_type ReturnType;
    auto task = [&range]{
        std::vector<std::shared_future<ReturnType>> fv;
        for(auto& t : range) {
            fv.emplace_back(t.run());
        }

        std::vector<ReturnType> v;
        for(auto& f : fv) {
            v.emplace_back(f.get());
        }
        return v;
    };
    return Task<std::vector<ReturnType>()>(task);
}

namespace detail {

//a trait
//use struct to get the type
//it help a lot when handle many different types template
template<typename R>
struct RangeTrait
{
    typedef R type;
};

template<typename R>
struct RangeTrait<std::shared_future<R>>
{
    typedef R type;
};

template<typename Range, typename Vec>
void transform(Range& range, Vec& fv) {
    // typedef typename Range::value_type::return_type ReturnType;
    for(auto& t : range) {
        fv.emplace_back(t.run());
    }
}

//input the future vector, return the pair which is running over
template<typename Range>
pair<int, typename RangeTrait<typename std::remove_reference<Range>::type::value_type>::type>
    get_any_result_pair(Range&& fv) {
    size_t size_ = fv.size();
    while(true) {
        for(size_t i = 0; i < size_; ++i) {
            if(fv[i].wait_for(std::chrono::milliseconds(1)) == std::future_status::ready) {
                return {i, fv[i].get()};
            }
        }
    }
}

} // detail

//the when_any function, just like the former when_all
//input a range stl type vector<Task<R()>>
//but it will return if one task is running over
//so it's output is Task<pair<size_t, R>()>, the index and its value
//same
//use get or run get the answer=> a pair which is running over and its value
template<typename Range, typename Vec>
Task<pair<int, typename Range::value_type::return_type>()> WhenAny(Range& range, Vec& fv) {
    typedef typename Range::value_type::return_type ReturnType;
    auto task = [&range, &fv] {
        using namespace detail;
        detail::transform(range, fv);
        return detail::get_any_result_pair(fv);
    };
    return Task<std::pair<int, typename Range::value_type::return_type>()>(task);
}

//caller need to input a vector of future, so give this type a name
//and let it easy to remember
template<typename T>
struct TaskFutVec
{
    typedef std::vector<std::shared_future<T>> type;
};

} // wzp


#endif // LIB_WHEN_ALL_ANY_HPP
