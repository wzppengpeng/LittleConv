#ifndef CMD_HPP
#define CMD_HPP

#include <stdio.h>

#include <string>
#include <vector>

#include "my_string/string.hpp"

#define MAXLINE 100

namespace wzp {

//the command exec function
inline static bool system(const std::string& cmd, std::vector<string>& cmd_result) {
    auto ptr = popen(cmd.c_str(), "r");
    if(ptr == nullptr) return false;
    char str[MAXLINE];
    while(fgets(str, MAXLINE, ptr)) {
        string cache(str);
        cmd_result.emplace_back(cache.substr(0, cache.size() - 1));
    }
    if(pclose(ptr) == -1) return false;
    return true;
}

} // wzp

#endif // CMD_HPP
