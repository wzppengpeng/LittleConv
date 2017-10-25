#ifndef LIB_TASK_GROUP_HPP
#define LIB_TASK_GROUP_HPP

#include <vector>
#include <future>
#include <memory>

#include "thread/task.hpp"

using std::vector;
namespace wzp {
//the group use a vector to save the task function object
//use run function to insert a task into group and let it run
//use wait to wait all tasks running over
//this group can only receive the task which the return type of them is void
class TaskGroup {
public:
    TaskGroup() = default;
    ~TaskGroup() {}

    /*the basic interface to receive task*/
    void run(Task<void()>& task) {
        m_group.emplace_back(task.run());
    }

    void run(Task<void()>&& task) {
        m_group.emplace_back(task.run());
    }

    template<typename F>
    void run(F&& f) {
        run(Task<void()>(std::forward<F>(f)));
    }

    template<typename First, typename... Funs>
    void run(First&& first, Funs&&... funs) {
        run(std::forward<First>(first));
        run(std::forward<Funs>(funs)...);
    }

    void run(std::vector<Task<void()>>& task_vec) {
        for(auto& task : task_vec) {
            run(task);
        }
    }

    void run(std::vector<Task<void()>>&& task_vec) {
        for(auto& task : task_vec) {
            run(task);
        }
    }

    void wait() {
        for(auto& task : m_group) {
            task.get();
        }
    }

    inline void reserve(size_t len) {
        m_group.reserve(len);
    }

private:
    std::vector<std::shared_future<void>> m_group;
};

using TaskGroupPtr = std::unique_ptr<TaskGroup>;

} // wzp


#endif // LIB_TASK_GROUP_HPP
