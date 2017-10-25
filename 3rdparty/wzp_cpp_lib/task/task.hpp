#ifndef TASK_HPP
#define TASK_HPP

#include <utility>
#include <functional>

namespace wzp {

//the list function class
template<typename T>
class Task;

template<typename R, typename... Args>
class Task<R(Args...)> {
public:
    Task(std::function<R(Args...)>&& f) : m_fn(std::move(f)) {}

    Task(std::function<R(Args...)>& f) : m_fn(f) {}

    //the run function
    template<typename... Args_>
    R run(Args_&&... args) {
        return m_fn(std::forward<Args_>(args)...);
    }

    //the then function
    template<typename F>
    auto then(F f)
    ->Task<typename std::result_of<F(R)>::type(Args...)> {
        return Task<typename std::result_of<F(R)>::type(Args...)>(
            [this, &f](Args&&... args) {
                return f(m_fn(std::forward<Args>(args)...));
            }
        );
    }

private:
    std::function<R(Args...)> m_fn;
};

} // wzp

#endif // TASK_HPP
