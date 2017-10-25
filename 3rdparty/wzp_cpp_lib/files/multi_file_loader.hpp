#ifndef MULTI_FILE_LOADER
#define MULTI_FILE_LOADER

/**
 * load big file by size as a iterator
 */
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

constexpr long long MAX_SIZE = 45905300000;

template<typename R>
class MultiFileLoader
{

typedef long long ll;

public:
    MultiFileLoader(ll chunk_size = MAX_SIZE,
        int workers = 8) : m_chunk_size(chunk_size),
        m_workers(workers),
        m_can_iter(true),
        m_location(0),
        m_iter_time(0)
    {}

    //the regis function
    //regis a function object and the filename
    template<typename Fun>
    void regis(Fun fn, const std::string& file) {
        m_fn = std::move(fn);
        m_filename = file;
        init(file);
    }

    //the iter rable
    inline bool has_next() const { return m_can_iter; }

    //iter function
    std::vector<R> next() {
        //set the pos for each thread
        auto pos_array = compute_position(m_chunk_size, m_location);
        std::vector<std::vector<R> > m_res_array(m_workers);
        ll end_loc = (m_location + m_chunk_size) < m_file_size ? m_location + m_chunk_size : m_file_size;
        wzp::ParallelRange(m_workers,
            [this, &pos_array, &m_res_array, end_loc](int pid) {
                process(pid, pos_array, m_filename, end_loc, m_res_array);
            });
        std::vector<R> res;
        flat(res, m_res_array);
        //post for one time iter
        post();
        return std::move(res);
    }

private:
    //the function object to handle the file
    std::function<R(const std::string&)> m_fn;
    //the loader handle size
    ll m_chunk_size;
    ll m_file_size;
    //the workers nnumber
    int m_workers;

    //params of iter
    bool m_can_iter;
    ll m_location;
    int m_iter_time;
    //the filename
    std::string m_filename;

private:
    //some inner function
    //get file max size
    ll get_file_size(const std::string& filename) {
        std::fstream f(filename.c_str(), std::fstream::ate|std::fstream::in);
        auto res = f.tellg();
        f.close();
        return res;
    }

    void init(const std::string& filename) {
        //set the file max size and the chunk size
        m_file_size = get_file_size(filename);
        m_chunk_size = std::min(m_chunk_size, m_file_size);
    }

    /**
     * Multi Thread Inner Functions, Copy From Multi Reader
     */
    std::vector<ll> compute_position(ll file_size_, ll begin) {
        ll file_size = (begin + file_size_) < m_file_size ? file_size_ : m_file_size - begin;
        std::vector<ll> pos_array(m_workers);
        auto block_size = file_size / m_workers;
        for(size_t i = 0; i < pos_array.size(); ++i) {
            pos_array[i] = i * block_size + begin;
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
        if(startpossition != m_location) {
            std::getline(f, buffer);
        }
        auto pos = f.tellg();
        std::vector<R> son_res_array;
        while(pos >= 0 && pos < endpossition) {
            std::getline(f, buffer);
            son_res_array.emplace_back(m_fn(buffer));
            pos = f.tellg();
        }
        if(pid == pos_array.size() - 1) m_location = pos;
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

    void update_location() {
        //read file read peek
        std::fstream f(m_filename, std::fstream::in);
        std::string buffer;
        f.seekg(m_location);
        std::getline(f, buffer);
        m_location = f.tellg();
    }

    void post() {
        ++m_iter_time;
        if(m_location >= m_file_size || m_location < 0) m_can_iter = false;
    }

};

} //wzp


#endif /*MULTI_FILE_LOADER*/