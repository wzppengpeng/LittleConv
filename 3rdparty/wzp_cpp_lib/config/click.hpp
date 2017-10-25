#ifndef ARG_CLICK_HPP_
#define ARG_CLICK_HPP_


#include <stdexcept>
#include <string>
#include <iostream>
#include <unordered_map>

#include "common.h"
#include "my_string/string.hpp"

using std::string;
using std::cout;
using std::cerr;
using std::endl;

namespace wzp
{


class Click
{

public:

    /**
     * the function to add args
     */
    inline static void add_argument(const string& key, const string& desc) {
        Get().emplace(key, std::make_pair("", desc));
    }

    //the right value type
    inline static void add_argument(string&& key, string&& desc) {
        Get().emplace(key, std::make_pair("", desc));
    }

    /**
     * the function to parse the args
     */
    static void parse(int argc, char** argv) noexcept {
        if(argc != static_cast<int>(Get().size() + 1)) {
            cerr << RED << "[Fatal] args size mismatch" << RESET << endl;
            help();
            throw std::logic_error("args numbers mismatch");
        }
        for(int i = 1; i < argc; ++i) {
            Parse(i, argv);
        }
    }

    /**
     * the functions to get key's values
     */
    static string get(const std::string& key) noexcept {
        auto it = Get().find(key);
        if(it == Get().end()) {
            cerr<<RED<<"[Fatal] the key is not exist: "<<key<<RESET<<std::endl;
            throw std::logic_error("Key Missing, check your code");
        }
        else {
            return it->second.first;
        }
    }

    //get the value by key, return other types
    template<typename T>
    static T get(const std::string& key) noexcept {
        auto it = Get().find(key);
        if(it == Get().end()) {
            cerr<<RED<<"[Fatal] the key is not exist: "<<key<<RESET<<std::endl;
            throw std::logic_error("Key Missing, check your code");
        }
        else {
            return convert_string<T>(it->second.first);
        }
    }

    /**
     * the help function
     */
    static void help() {
        for(auto& p : Get()) {
            cout << "Key: " << YELLOW << p.first << RESET << "   Describes: " << GREEN
                << p.second.second << RESET << std::endl;
        }
    }

    /**
     * the function to get the local dict of args
     */
    static std::unordered_map<string, std::pair<string, string> >& Get() {
        static std::unordered_map<string, std::pair<string, string> > args_map;
        return args_map;
    }

private:
    static void Parse(int index, char** argv) {
        string cache(argv[index]);
        auto pos = cache.find('=');
        if(pos == string::npos) {
            cerr << RED << "[Fatal] the Args is Set Not Right, USAGE: arg=xx" << RESET << std::endl;
            help();
            throw std::logic_error("the args are not right");
        }
        auto key = std::move(cache.substr(0, pos));
        trim(key);
        auto value = std::move(cache.substr(pos+ 1));
        trim(value);
        auto it = Get().find(key);
        if(it == Get().end()) {
            cerr << RED << "[Fatal] key not exist " << key << RESET << std::endl;
            help();
            throw std::logic_error("miss key");
        }
        it->second.first = std::move(value);
    }

};


} //wzp


#endif /*ARG_CLICK_HPP_*/