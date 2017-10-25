#ifndef LIB_PARALLEL_ALGORITHM_HPP
#define LIB_PARALLEL_ALGORITHM_HPP

#include <algorithm>
#include <future>
#include <vector>

#include "thread/task_group.hpp"

namespace wzp {

/**
 * ParallelRange:input range i to len, and a function, the function need receive the index
 */
template<typename Index, typename Function>
void ParallelRange(Index begin, Index end, Function&& fun) {
    auto thread_num = std::thread::hardware_concurrency();
    auto block_size = (end - begin) / thread_num;
    TaskGroup group;
    Index last = begin;
    if(block_size) {
        last += (thread_num - 1) * block_size;
    }
    else {
        last = end;
        block_size = 1;
    }
    group.reserve(thread_num + 1);
    for(; begin < last; begin += block_size) {
        group.run([&fun, begin, block_size] () {
            for(Index i = begin; i < begin + block_size; ++i) {
                fun(i);
            }
        });
    }
    //the last
    if(begin < end) {
        group.run([&fun, begin, end] () {
            for(Index i = begin; i < end; ++i) {
                fun(i);
            }
        });
    }
    group.wait();
}

/**
 * the simple one
 */
template<typename Index, typename Function>
void ParallelRange(Index end, Function&& fun) {
    ParallelRange(Index(0), end, fun);
}

//use the hardware thread to parallel handle the fun
template<typename Iterator, typename Function>
void ParallelForeach(Iterator begin, Iterator end, Function&& fun) {
    auto thread_num = std::thread::hardware_concurrency();
    auto block_size = std::distance(begin, end) / thread_num;
    Iterator last = begin;
    if(block_size) {
        std::advance(last, (thread_num - 1) * block_size);
    }
    else {
        last = end;
        block_size = 1;
    }
    std::vector<std::future<void>> futures;
    futures.reserve(thread_num + 1);
    //the first
    for(; begin != last; std::advance(begin, block_size)) {
        futures.emplace_back(std::async(std::launch::async,
        [begin, block_size, &fun] {
            std::for_each(begin, begin + block_size, fun);
        }
        ));
    }
    //the last thread
    futures.emplace_back(std::async(std::launch::async, [begin, end, &fun]{
        std::for_each(begin, end, fun);
    }));
    for(auto& fut : futures) {
        fut.get();
    }
}

//use the task group, give it a list of functions
//use a task group tu run these functions
//these funs need to be return void
template<typename... Funs>
void ParallelInvoke(Funs&&... rest)
{
    TaskGroup group;
    group.run(std::forward<Funs>(rest)...);
    group.wait();
}

//the map-reduce function
//just like the pre parallel foreach function, foreach is for range
//which don't need answer as they are independent
//ParallelReduce function is for which not depentent each other
//so map in different thread, then reduce to get answers
template<typename Range, typename RangeFunc, typename ReduceFunc>
inline typename std::remove_reference<Range>::type::value_type ParallelReduce(Range&& range,
    typename std::remove_reference<Range>::type::value_type init, RangeFunc&& range_func, ReduceFunc&& reduce_func) {
    auto thread_num = std::thread::hardware_concurrency();
    auto begin = std::begin(std::forward<Range>(range));
    auto end = std::end(std::forward<Range>(range));
    auto block_size = std::distance(begin, end) / thread_num;
    typename std::remove_reference<Range>::type::iterator last = begin;
    if(block_size) {
        std::advance(last, (thread_num - 1) * block_size);
    }
    else {
        last = end;
        block_size = 1;
    }
    typedef typename std::remove_reference<Range>::type::value_type ValueType;
    std::vector<std::future<ValueType>> futures;
    futures.reserve(thread_num + 1);
    //first thread_num - 1 thread
    for(; begin != last; std::advance(begin, block_size)) {
        futures.emplace_back(std::async(std::launch::async,
        [begin, &init, block_size, &range_func] {
            return range_func(begin, begin + block_size, init);
        }
        ));
    }
    //last thread
    futures.emplace_back(std::async(std::launch::async,
        [begin, end, &init, &range_func] {
            return range_func(begin, end, init);
    }));

    //get results
    std::vector<ValueType> results;
    results.reserve(thread_num + 1);
    for(auto& fut : futures) {
        results.emplace_back(fut.get());
    }

    return reduce_func(std::begin(results), std::end(results), init);
}

//the reduce function and range function is the same type
//most situations is like this
template<typename Range, typename ReduceFunc>
inline typename std::remove_reference<Range>::type::value_type ParallelReduce(Range&& range,
    typename std::remove_reference<Range>::type::value_type init, ReduceFunc&& reduce_func) {
    return ParallelReduce<Range, ReduceFunc>(std::forward<Range>(range),
        std::move(init),
        std::forward<ReduceFunc>(reduce_func),
        std::forward<ReduceFunc>(reduce_func));
}

} // wzp

#endif /*LIB_PARALLEL_ALGORITHM_HPP*/
