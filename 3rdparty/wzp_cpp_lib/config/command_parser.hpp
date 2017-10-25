#ifndef COMMAND_PARSER_HPP
#define COMMAND_PARSER_HPP

#include <string>
#include <sstream>
#include <unordered_map>

#include "log/log.hpp"
#include "function/help_function.hpp"


namespace wzp {

class CommandParser
{
public:
    using ValueDescrib = std::pair<std::string, std::string>;

    //define the needed params
    static void define_param(std::string&& key, std::string&& describle) {
        get_parser().m_params_map[key] = std::move(ValueDescrib({"", describle}));
    }

    //set the argc and argv
    static void set_args(int argc, char** argv) {
        get_parser().m_argc = argc;
        get_parser().m_argv = argv;
        get_parser().parse();
    }

    //the params get
    template<typename T>
    static bool get_param(std::string&& key, T& val) {
        std::string real_key = "--" + key;
        if(get_parser().m_params_map.find(real_key) == get_parser().m_params_map.end()) {
            get_parser().print_help();
            return false;
        }
        std::string temp = get_parser().m_params_map[real_key].first;
        std::stringstream ss;
        ss<<temp;
        ss>>val;
        return true;
    }

    static bool get_param(std::string&& key, std::string& val) {
        std::string real_key = "--" + key;
        if(get_parser().m_params_map.find(real_key) == get_parser().m_params_map.end()) {
            get_parser().print_help();
            return false;
        }
        val = get_parser().m_params_map[real_key].first;
        return true;
    }

private:
    //the single mode use
    CommandParser() = default;
    CommandParser(const CommandParser&) = delete;
    CommandParser(CommandParser&&) = delete;

    //the static method for out to get the instance
    static CommandParser& get_parser() {
        static CommandParser parser;
        return parser;
    }

    //parse the command and put them into the map
    void parse() {
        if(get_parser().m_argc % 2 == 0) {
            get_parser().print_help();
            log::fatal("the input number of params is not right");
        }
        for(auto i = 1; i <= get_parser().m_argc - 2; i+=2) {
            if(get_parser().m_params_map.find(m_argv[i]) == get_parser().m_params_map.end()) {
                get_parser().print_help();
                log::fatal("the input args is not the right type");
            }
            get_parser().m_params_map[m_argv[i]].first = m_argv[i + 1];
        }
    }

    void print_help() {
        print("USAGE:");
        for(auto& p : get_parser().m_params_map) {
            print(p.first, ":", p.second.second);
        }
    }

private:
    //the map to store the params
    std::unordered_map<std::string, ValueDescrib> m_params_map;
    //the input char args
    int m_argc;
    char** m_argv;
};

} //wzp


#endif // COMMAND_PARSER_HPP
