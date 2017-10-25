#ifndef REDIS_TASK_LISTEN_HPP
#define REDIS_TASK_LISTEN_HPP


#include <functional>
#include <atomic>
#include <memory>

#include "redis/redis_oper.hpp"

namespace wzp {

class RedisTaskListen
{
public:
    RedisTaskListen(const std::string& ip, int port, const std::string& list_name) :
        done(false),
        m_ip(ip),
        m_port(port),
        m_list_name(list_name),
        oper_ptr(nullptr)
        {
            init();
        }

    ~RedisTaskListen(){}

    //set the handler to run the loop listen task
    template<typename Fun>
    void set_handler(Fun f) {
        m_handler = f;
    }

    //the main tunction to run the loop listen task
    void run(bool is_prioty = false) {
        while(!done) {
            std::string res;
            if(!is_prioty)
                res = oper_ptr->blpop(m_list_name);
            else
                res = oper_ptr->brpop(m_list_name);
            if(!res.empty() && m_handler)
                m_handler(res);
        }
    }


public:
    //the loop control thread safe
    std::atomic<bool> done;

private:
    //create the oper_ptr object
    inline void init() {
        oper_ptr = RedisOperPtr(new RedisOper(m_ip, m_port));
    }

private:
    //the listen redis location
    std::string m_ip;
    int m_port;
    std::string m_list_name;

    //the redis oper
    RedisOperPtr oper_ptr;

    //the handler to do the task, when listen out the redis list
    std::function<void(const std::string&)> m_handler;

};

} // wzp

using RedisTaskListenPtr = std::unique_ptr<wzp::RedisTaskListen>;

#endif // REDIS_TASK_LISTEN_HPP
