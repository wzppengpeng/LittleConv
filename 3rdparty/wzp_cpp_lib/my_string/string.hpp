#ifndef MY_STRING_HPP_
#define MY_STRING_HPP_

#include <string>
#include <sstream>
#include <vector>

namespace wzp {

/*convert string to any type, this is too old, please use lexical_cast<string> instead*/
template<typename T>
inline static T convert_string(const std::string& input) {
    std::istringstream ss(input);
    T output;
    ss >> output;
    return output;
}

template<typename T>
inline static std::string convert_to_string(T&& t) {
    std::stringstream ss;
    ss << t;
    std::string res;
    ss >> res;
    return res;
}


/*the specific one to split strings into vector<string>*/
inline static std::vector<std::string> split_string(const std::string& input, char split_delemeter) {
    std::vector<std::string> result;
    if (input.empty()) return std::move(result);
    std::istringstream ss(input);
    std::string cache;
    while (std::getline(ss, cache, split_delemeter)) {
        result.emplace_back(cache);
    }
    return std::move(result);
}

/**
 * transform vector<string> to other type data
 */
template<typename T>
inline static std::vector<T> transform(const std::vector<std::string>& strs) {
    std::vector<T> datas(strs.size());
    for(size_t i = 0; i < strs.size(); ++i) {
        datas[i] = convert_string<T>(strs[i]);
    }
    return std::move(datas);
}

/**
 * the combine type
 */
template<typename T>
inline static std::vector<T> split_string_transform(const std::string& input, char split_delemeter) {
    std::vector<T> result;
    if (input.empty()) return std::move(result);
    std::istringstream ss(input);
    std::string cache;
    while (std::getline(ss, cache, split_delemeter)) {
        result.emplace_back(convert_string<T>(cache));
    }
    return std::move(result);
}

/*join a char into vector<string>, and get the string*/

/**
 * join a char into a vector<T>, get the string
 */
template<typename T>
inline static std::string join_string(const std::vector<T>& datas, char delimiter) {
    if(datas.empty()) return "";
    std::stringstream ss;
    ss << datas[0];
    for (size_t i = 1; i < datas.size(); ++i) {
        ss << delimiter;
        ss << datas[i];
    }
    return std::move(ss.str());
}


/*trim funciton*/
inline static bool trim(std::string& str) {
    if (str.empty()) return false;
    str.erase(str.find_last_not_of(" \f\t\n\r\v") + 1);
    str.erase(0, str.find_first_not_of(" \f\t\n\r\v"));
    return true;
}

/*the string format*/
template<typename T>
inline static std::string format(std::string&& str, T&& t) {
    auto begin = str.find_first_of('{');
    auto end = str.find_first_of('}');
    return std::move(str.substr(0, begin) + wzp::convert_to_string(std::forward<T>(t))
     + str.substr(end + 1));
}

// /*the string format with any params*/
template<typename T, typename... Args>
inline static std::string format(std::string&& str, T&& t, Args&&... args) {
    auto begin = str.find_first_of('{');
    auto end = str.find_first_of('}');
    return std::move(format(std::move(str.substr(0, begin) + wzp::convert_to_string(std::forward<T>(t))
     + str.substr(end + 1)), std::forward<Args>(args)...));
}

/**
 * the startwith function
 */
inline static bool start_with(const std::string& s, const std::string& p) {
    if(p.size() > s.size()) return false;
    for(size_t i = 0; i < p.size(); ++i) {
        if(p[i] != s[i]) return false;
    }
    return true;
}

/**
 * the end with function
 */
inline static bool end_with(const std::string& s, const std::string& p) {
    if(p.size() > s.size()) return false;
    auto len_s = s.size();
    auto len_j = p.size();
    for(size_t j = 0; j < len_j; ++j) {
        if(p[len_j - 1 - j] != s[len_s - 1 - j]) return false;
    }
    return true;
}

}

#endif
