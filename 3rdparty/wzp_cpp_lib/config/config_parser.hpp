#ifndef CONFIG_PARSER_HPP
#define CONFIG_PARSER_HPP

#include <fstream>
#include <string>
#include <iostream>
#include <map>

#include "my_string/string.hpp"

using std::string;
using std::cerr;
using std::cout;
using std::endl;

namespace wzp {

class ConfigParser
{
public:
    ConfigParser() : m_config_path(),
                     m_equal_symbol("="),
                     m_comment_symbol("#"),
                     m_params_map()
    {
    }

    ConfigParser(const std::string& config) : m_config_path(config),
                                              m_equal_symbol("="),
                                              m_comment_symbol("#"),
                                              m_params_map()
    {}

    ~ConfigParser(){}

    bool config() { return parse(); }

    bool get(const std::string& key, string& val) {
        std::map<std::string, std::string>::iterator it = m_params_map.find(key);
        if(it == m_params_map.end()) return false;
        else {
            val = it->second;
            return true;
        }
    }

    template<typename T>
    bool get(const std::string& key, T& val) {
        std::map<std::string, std::string>::iterator it = m_params_map.find(key);
        if(it == m_params_map.end()) return false;
        else {
            val = convert_string<T>(it->second);
            return true;
        }
    }

    /**
     * update config
     * @param std::string key, value
     */
    void update(const std::string& key, const std::string& value) {
        std::map<std::string, std::string>::iterator it = m_params_map.find(key);
        if(it != m_params_map.end()) {
            it->second = value;
        }
        else {
            m_params_map.insert(std::pair<std::string, std::string>(key, value));
        }
    }

private:
    bool parse() {
        std::ifstream input(m_config_path.c_str(), std::ios::in);
        if(!input) {
            cerr<<"[fatal] cannot open the config file"<<endl;
            return false;
        }
        string line;
        while(std::getline(input, line)) {
            trim(line);
            size_t comment_pos = line.find(m_comment_symbol);
            if(comment_pos != string::npos) {
                line = line.substr(0, comment_pos);
            }
            size_t pos = line.find(m_equal_symbol);
            if(pos == string::npos) continue;
            string key(line.substr(0, pos));
            string value(line.substr(pos + 1));
            trim(key);
            trim(value);
            m_params_map.emplace(key, value);
        }
        input.close();
        return true;
    }

private:
    std::string m_config_path;
    std::string m_equal_symbol; // default = "="
    std::string m_comment_symbol; // default is "#"

    //the config comtainer
    std::map<std::string, std::string> m_params_map;
};


} // wzp


#endif // CONFIG_PARSER_HPP
