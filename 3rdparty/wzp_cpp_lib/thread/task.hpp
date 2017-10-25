#ifndef LIB_TASK_HPP
#define LIB_TASK_HPP

#include <utility>
#include <functional>
#include <future>
#include <thread>

namespace wzp {

//declare this is a template type
template<typename T>
class Task;

template<typename R, typename... Args>
class Task<R(Args...)> {
public:
    //the return type for others to use
    typedef R return_type;
    //R is the return type
    //the constructor
    Task(std::function<R(Args...)>&& f) : m_fn(std::move(f)) {}

    Task(std::function<R(Args...)>& f) : m_fn(f) {}

    ~Task() {}

    //wait interface
    //std::launch::async
    void wait() {
        std::async(m_fn).wait();
    }

    //get interface
    template<typename... A>
    R get(A&&... a) {
        return std::async(m_fn, std::forward<A>(a)...).get();
    }

    //run interface, return the future, the basic one
    //it will async run in real type, just not block the main thread
    template<typename... A>
    std::shared_future<R> run(A&&... a) {
        return std::async(std::launch::async, m_fn, std::forward<A>(a)...);
    }

    //the then function, as a list call
    //give in a new function, then return a new task
    //the pre one return is the post one's input
    template<typename F>
    auto then(F&& f)
    ->Task<typename std::result_of<F(R)>::type(Args...)> {
        typedef typename std::result_of<F(R)>::type ReturnType;
        auto func = std::move(m_fn);
        return Task<ReturnType(Args...)>(
            [func, &f](Args&&... args) {
                std::future<R> lastf = std::async(func, std::forward<Args>(args)...);
                return std::async(f, lastf.get()).get();
            }
        );
    }

private:
    std::function<R(Args...)> m_fn;//R is the Return type
};

//the type of void return type
template<typename... Args>
class Task<void(Args...)> {
public:
    //void is the return type
    //the constructor
    Task(std::function<void(Args...)>&& f) : m_fn(std::move(f)) {}

    Task(std::function<void(Args...)>& f) : m_fn(f) {}

    ~Task() {}

    //wait interface
    //std::launch::async
    void wait() {
        std::async(m_fn).wait();
    }
    //run interface, return the future, the basic one
    //it will async run in real type, just not block the main thread
    template<typename... A>
    std::shared_future<void> run(A&&... a) {
        return std::async(std::launch::async, m_fn, std::forward<A>(a)...);
    }

    template<typename... A>
    void get(A&&... a) {
        std::async(m_fn, std::forward<A>(a)...).get();
    }

private:
    std::function<void(Args...)> m_fn;//void is the Return type
};

} // wzp

#endif // LIB_TASK_HPP
