#ifndef REDIS_OPER_HPP_
#define REDIS_OPER_HPP_

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <hiredis/hiredis.h>

#include "my_string/string.hpp"

namespace wzp {

class RedisOper {
public:
    //constructor, give the ip and port
    RedisOper() = default;

    RedisOper(const std::string& ip, int port) : m_ip(ip), m_port(port)
    {
        //connect the redis
        if(connect_redis()) {
            m_is_connected = true;
        }
    }

    ~RedisOper() {
        shut_down();
    }

    //the connect function
    bool connect_redis() {
        redis = redisConnect(m_ip.c_str(), m_port);
        if(redis == nullptr || redis->err) {
            redisFree(redis);
            return false;
        }
        m_is_connected = true;
        return true;
    }

    /*get if the redis is connected*/
    bool is_connected() const {
        return m_is_connected;
    }

    /*set the key = value*/
    template<typename Val_t>
    bool set(const std::string& key, Val_t&& t) {
        if(!m_is_connected)
            return false;
        std::string cmd = "set {} {}";
        cmd = wzp::format(std::move(cmd), key, std::forward<Val_t>(t));
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return false;
        }
        else {
            freeReplyObject(reply);
            return true;
        }
    }

    /*get the key's value*/
    std::string get(const std::string& key) {
        if(!m_is_connected)
            return "";
        std::string cmd = "get {}";
        cmd = wzp::format(std::move(cmd), key);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return "";
        }
        else {
            std::string res(reply->str, reply->len);
            freeReplyObject(reply);
            return res;
        }
    }

    /*rpush something into the queue*/
    template<typename Val_t>
    bool rpush(const std::string& key, Val_t&& t) {
        if(!m_is_connected)
            return false;
        std::string cmd = "rpush {} {}";
        cmd = wzp::format(std::move(cmd), key, std::forward<Val_t>(t));
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return false;
        }
        else {
            freeReplyObject(reply);
            return true;
        }
    }

    /*lpop a value from the queue*/
    std::string lpop(const std::string& key) {
        if(!m_is_connected) {
            return "";
        }
        std::string cmd {"lpop {}"};
        cmd = wzp::format(std::move(cmd), key);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return "";
        }
        else {
            std::string res(reply->str, reply->len);
            freeReplyObject(reply);
            return res;
        }
    }

    /*lrange : get a list valus the begin and length to get*/

    /*the block lpop function*/
    std::string blpop(const std::string& key, int time_out = 100) {
        if(!m_is_connected) {
            return "";
        }
        std::string cmd {"blpop {} {}"};
        cmd = wzp::format(std::move(cmd), key, time_out);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return "";
        }
        else {
            if(reply->elements == 0) return "";
            auto ptr = reply->element[1];
            std::string res(ptr->str, ptr->len);
            freeReplyObject(reply);
            return res;
        }
    }

    //the block rpop function
    std::string brpop(const std::string& key, int time_out = 100) {
        if(!m_is_connected) {
            return "";
        }
        std::string cmd {"brpop {} {}"};
        cmd = wzp::format(std::move(cmd), key, time_out);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return "";
        }
        else {
            if(reply->elements == 0) return "";
            auto ptr = reply->element[1];
            std::string res(ptr->str, ptr->len);
            freeReplyObject(reply);
            return res;
        }
    }

    /*get the length of a list*/
    long long llen(const std::string& key) {
        if(!m_is_connected) return -1;
        std::string cmd {"llen {}"};
        cmd = wzp::format(std::move(cmd), key);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return -1;
        }
        else {
            long long res = reply->integer;
            freeReplyObject(reply);
            return res;
        }
    }

    /*hash_set give the key name with key and value*/
    template<typename Val_t>
    bool hash_set(const std::string& key_name, const std::string& key, Val_t&& t) {
        if(!m_is_connected)
            return false;
        std::string cmd {"HSET {} {} {}"};
        cmd = wzp::format(std::move(cmd), key_name, key, std::forward<Val_t>(t));
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return false;
        }
        else {
            freeReplyObject(reply);
            return true;
        }
    }

    /*hash get a keyname's key, return the value*/
    std::string hash_get(const std::string& key_name, const std::string& key) {
        if(!m_is_connected)
            return "";
        std::string cmd {"hget {} {}"};
        cmd = wzp::format(std::move(cmd), key_name, key);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return "";
        }
        else {
            std::string res(reply->str, reply->len);
            freeReplyObject(reply);
            return res;
        }
    }

    /*delete the hash keyname's key*/
    bool hash_delete(const std::string& key_name, const std::string& key) {
        if(!m_is_connected)
            return false;
        std::string cmd {"HDEL {} {}"};
        cmd = wzp::format(std::move(cmd), key_name, key);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return false;
        }
        else {
            freeReplyObject(reply);
            return true;
        }
    }

    /*delete a key in redis*/
    bool del(const std::string& key) {
        if(!m_is_connected)
            return false;
        std::string cmd {"DEL {}"};
        cmd = wzp::format(std::move(cmd), key);
        redisReply* reply = (redisReply*)redisCommand(redis, cmd.c_str());
        if(reply->type == REDIS_REPLY_ERROR) {
            freeReplyObject(reply);
            return false;
        }
        else {
            freeReplyObject(reply);
            return true;
        }
    }

private:

    void shut_down() {
        redisFree(redis);
        redis = nullptr;
    }

private:
    std::string m_ip = "127.0.0.1";
    int m_port = 6379;

    bool m_is_connected = false;
    //the redis connect ptr
    redisContext* redis = nullptr;
};

} //wzp

using RedisOperPtr = std::unique_ptr<wzp::RedisOper>;

#endif /*REDIS_OPER_HPP_*/
