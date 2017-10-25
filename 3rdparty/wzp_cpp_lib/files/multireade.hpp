#ifndef MULTIREADER_HPP_
#define MULTIREADER_HPP_

#include <fstream>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>

#include "thread/parallel_algorithm.hpp"

#include "function/help_function.hpp"

namespace wzp
{

using wzp::print;

template<typename R>
class MultiReader
{

typedef long long ll;

public:
    MultiReader(int w = 8) : m_workers(w) {}

    //the register function
    template<typename Fun>
    void regis(Fun f) {
        m_fn = f;
    }

    /**
     * get a big file's size
     * @param  filename string
     * @return          size
     */
    long long get_file_size(const std::string& filename) {
        std::fstream f(filename.c_str(), std::fstream::ate|std::fstream::in);
        auto res = f.tellg();
        f.close();
        return res;
    }

    //the run function
    std::vector<R> run(const std::string& file) {
        auto file_size = get_file_size(file);
        auto pos_array = compute_position(file_size);
        std::vector<std::vector<R> > m_res_array(m_workers);
        wzp::ParallelRange(m_workers,
            [this, &pos_array, file_size, &file, &m_res_array](int pid) {
                process(pid, pos_array, file, file_size, m_res_array);
            });
        std::vector<R> res;
        flat(res, m_res_array);
        return std::move(res);
    }

private:
    //the function object to handle the line
    std::function<R(const std::string&)> m_fn;

    //the workers num
    int m_workers;

private:
    std::vector<ll> compute_position(ll file_size) {
        std::vector<ll> pos_array(m_workers);
        auto block_size = file_size / m_workers;
        for(size_t i = 0; i < pos_array.size(); ++i) {
            pos_array[i] = i * block_size;
        }
        return std::move(pos_array);
    }

    //the process function
    void process(size_t pid, const std::vector<ll>& pos_array, const std::string& file,
        ll file_size, std::vector<std::vector<R> >& m_res_array) {
        std::fstream f(file, std::fstream::in);
        auto startpossition = pos_array[pid];
        auto endpossition = (pid == pos_array.size() - 1) ? file_size : pos_array[pid + 1];
        std::string buffer;
        f.seekg(startpossition);
        if(startpossition != 0) {
            std::getline(f, buffer);
        }
        auto pos = f.tellg();
        std::vector<R> son_res_array;
        while(pos >= 0 && pos < endpossition) {
            std::getline(f, buffer);
            son_res_array.emplace_back(m_fn(buffer));
            pos = f.tellg();
        }
        f.close();
        m_res_array[pid] = std::move(son_res_array);
    }

    void flat(std::vector<R>& res, std::vector<std::vector<R> >& m_res_array) {
        // res.reserve(m_workers * m_res_array.front().size());
        for(auto& res_array : m_res_array) {
            std::copy(std::begin(res_array), std::end(res_array), std::back_inserter(res));
            //clear the memory
            vector<R>().swap(res_array);
        }
    }

};


} //wzp


#endif /*MULTIREADER_HPP_*/