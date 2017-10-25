#ifndef ARG_PARSER_HPP_
#define ARG_PARSER_HPP_

#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>


#include "common.h"
#include "my_string/string.hpp"

using std::string;
using std::cerr;
using std::endl;

namespace wzp {

class ArgParser {
public:

    //get the args and parse them
    static void parse(int argc, char** argv) noexcept {
        for(int i = 1; i < argc; ++i) {
            Parse(i, argv);
        }
    }

    //get the value by key
    static string get(const std::string& key) noexcept {
        auto it = Get().find(key);
        if(it == Get().end()) {
            cerr<<RED<<"[Fatal] the key is not exist: "<<key<<RESET<<std::endl;
            throw std::logic_error("Key Missing");
        }
        else {
            return it->second;
        }
    }

    //get the value by key, return other types
    template<typename T>
    static T get(const std::string& key) noexcept {
        auto it = Get().find(key);
        if(it == Get().end()) {
            cerr<<RED<<"[Fatal] the key is not exist: "<<key<<RESET<<std::endl;
            throw std::logic_error("Key Missing");
        }
        else {
            return convert_string<T>(it->second);
        }
    }

    //local static function to get the container to store the args
    static std::unordered_map<string, string>& Get() {
        static std::unordered_map<string, string> args_map;
        return args_map;
    }

private:
    inline static void Parse(int index, char** argv) {
        string cache(argv[index]);
        auto pos = cache.find('=');
        if(pos == string::npos) {
            cerr<<RED<<"[Fatal] the Args is Set Not Right, USAGE: arg=xx"<<RESET<<std::endl;
            throw std::logic_error("the args is not right");
        }
        auto key = std::move(cache.substr(0, pos));
        trim(key);
        auto value = std::move(cache.substr(pos+ 1));
        trim(value);
        Get().emplace(std::move(key), std::move(value));
    }

};


} //wzp

#endif /*ARG_PARSER_HPP_*/