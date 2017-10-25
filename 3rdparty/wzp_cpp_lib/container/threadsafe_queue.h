#ifndef THREAD_SAFE_QUEUE_H_
#define THREAD_SAFE_QUEUE_H_

#include <mutex>
#include <condition_variable>
#include <deque>
#include <memory>
#include <vector>

namespace wzp {
template<typename T>
class threadsafe_queue
{
private:
    mutable std::mutex mut;
    std::deque<T> data_queue;
    std::condition_variable data_cond;

public:
    enum class priority {
        HIGH,
        LOW
    };

public:
    threadsafe_queue() {}
    ~threadsafe_queue() {}

    /*push data into the queue by priority
    input:@new_value is the task or something new
    @pri is the priority by this class define
    return none
    */
    void push(T new_value, priority pri = priority::LOW) {
        std::lock_guard<std::mutex> lk(mut);
        if (pri == priority::HIGH) {
            data_queue.emplace_front(std::move(new_value));
        }
        else {
            data_queue.emplace_back(std::move(new_value));
        }
        data_cond.notify_one();
    }

    /*wait the queue if it is empty and pop one
    input:@ value is the reference of the output value
    */
    void wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] {return !data_queue.empty();});
        value = std::move(data_queue.front());
        data_queue.pop_front();
    }

    /*return the shared_ptr*/
    std::shared_ptr<T> wait_and_pop() {
        std::unique_lock<std::mutex> lk(mut);
        data_cond.wait(lk, [this] {return !data_queue.empty();});
        auto res = std::make_shared<T>(std::move(data_queue.front()));
        return res;
    }

    /*try pop if the queue is empty return false
    input:@value is the reference of the output value
    */
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty()) return false;
        value = std::move(data_queue.front());
        data_queue.pop_front();
        return true;
    }

    /*return the shared_ptr*/
    std::shared_ptr<T> try_pop() {
        std::lock_guard<std::mutex> lk(mut);
        if (data_queue.empty())
            return std::shared_ptr<T>();
        auto res = std::make_shared<T>(std::move(data_queue.front()));
        data_queue.pop_front();
        return res;
    }

    /*judge the queue is not empty*/
    bool empty() const {
        std::lock_guard<std::mutex> lk(mut);
        return data_queue.empty();
    }

};
}
#endif /*THREAD_SAFE_QUEUE_H_*/