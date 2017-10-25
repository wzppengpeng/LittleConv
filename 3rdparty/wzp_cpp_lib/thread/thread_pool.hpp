#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <thread>
#include <atomic>
#include <future>
#include <functional>
#include <vector>
#include <utility>
#include <tuple>

#include "container/threadsafe_queue.h"

#include "function/apply_tuple.hpp"


namespace wzp {
/*manage the threads(vector<thread>)*/
class join_threads {
private:
	std::vector<std::thread>& threads;
public:
	explicit join_threads(std::vector <std::thread>& threads_) :threads(threads_) {}
	~join_threads() {
		decltype(threads.size()) i = 0;
		for (; i < threads.size(); ++i) {
			if (threads[i].joinable()) {
				threads[i].join();
			}
		}
	}
};

/*the thread pool class*/
class thread_pool {
private:
	using Task = std::function<void()>;
	std::atomic<bool> done;
	threadsafe_queue<Task> task_queue;
	std::vector<std::thread> threads;
	join_threads join_threads_;
private:
	/*pop one task or sleep for each thread*/
	void work_thread() {
		while (!done) {
			Task task;
			task_queue.wait_and_pop(task);
			task();
		}
	}

public:
	thread_pool(const int pool_num = -1)
		:done(false),
		threads(),
		join_threads_(threads) {
		auto num = 0;
		if (pool_num == -1) {
			num = std::thread::hardware_concurrency();
		}
		else {
			num = pool_num;
		}
		try {
			for (auto i = 0; i < num; ++i) {
				threads.emplace_back(&thread_pool::work_thread, this);
			}
		}
		catch (...) {
			done = true;
			throw;
		}
	}

	~thread_pool(){
		done = true;
	}

	template<typename F>
	void submit(F&& f) {
		task_queue.push(Task(f));
	}

	void close() {
		done = true;
	}

	//give a loop interface
	void loop() {
		for (size_t i = 0; i < threads.size(); ++i) {
			if (threads[i].joinable()) {
				threads[i].join();
			}
		}
	}

};


/**
 * the thread pool with args
 */
template<typename... Args>
class ThreadPool
{

private:
	using Task = std::function<void(Args...)>;
	std::atomic<bool> m_done;
	threadsafe_queue<std::pair<Task, std::tuple<Args...> > > m_task_queue;
	std::vector<std::thread> m_threads;
	join_threads m_join_threads_;

private:
	/**
	 * pop one task and its args to run
	 */
	void work_thread() {
		while(!m_done) {
			std::pair<Task, std::tuple<Args...> > task;
			m_task_queue.wait_and_pop(task);
			wzp::apply(task.first, task.second);
		}
	}

public:
	ThreadPool(const int thread_num = -1)
	: m_done(false),
	m_threads(),
	m_join_threads_(m_threads) {
		int num = 0;
		if (thread_num < 0) num = std::thread::hardware_concurrency();
		else num = thread_num;
		try {
			for(int i = 0; i < num; ++i) {
				m_threads.emplace_back(&ThreadPool::work_thread, this);
			}
		} catch (...) {
			m_done = true;
			throw;
		}
	}

	~ThreadPool() { m_done = true; }

	//the interface to submit task and args
	template<typename Fun>
	void submit(Fun&& f, const std::tuple<Args...>& t) {
		m_task_queue.push(std::make_pair(Task(std::forward<Fun>(f)),
			t));
	}

	void loop() {
        for (size_t i = 0; i < m_threads.size(); ++i) {
            if (m_threads[i].joinable()) {
                m_threads[i].join();
            }
        }
    }

    inline void close() { m_done = true; }

};



} //wzp


#endif /*THREAD_POOL_H_*/
