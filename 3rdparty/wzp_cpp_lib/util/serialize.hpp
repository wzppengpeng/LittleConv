#ifndef SERIALIZE_HPP_
#define SERIALIZE_HPP_

#include <string>
#include <vector>
#include <fstream>

using std::vector;
using std::string;

namespace wzp
{

template<typename T>
/**
 * [serialize one data into a string cache]
 * @param cache the point of string
 * @param t     the data need to be serialize
 */
void serialize(std::string* cache, T& t) {
    string temp;
    temp.resize(sizeof(T));
    auto work_ptr = reinterpret_cast<char*>(&t);
    for (int i = 0; i < sizeof(T); ++i)
    {
        temp[i] = work_ptr[i];
    }
    cache->append(std::move(temp));
}

template<typename Dtype>
/**
 * [serialize vector data]
 * @param cache the point of cache string
 * @param v     the Dtype vector
 */
void serialize(std::string* cache, vector<Dtype>& v) {
    string temp;
    temp.resize(v.size() * sizeof(Dtype));
    auto work_ptr = reinterpret_cast<char*>(&v[0]);
    for (int i = 0; i < temp.size(); ++i)
    {
        temp[i] = work_ptr[i];
    }
    cache->append(std::move(temp));
}

template<typename T>
/**
 * serialize the point
 */
void serialize(std::string* cache, T* v, size_t len) {
    string temp;
    temp.resize(len * sizeof(T));
    T* c = v;
    auto work_ptr = reinterpret_cast<char*>(c);
    for (int i = 0; i < temp.size(); ++i)
    {
        temp[i] = work_ptr[i];
    }
    cache->append(std::move(temp));
}

template<typename T, typename... Rest>
/**
 * [serialize description]
 * @param cache point of string
 * @param t     the one data
 * @param rest  the rest of template data
 */
void serialize(std::string* cache, T& t, Rest&... rest) {
    serialize<T>(cache, t);
    serialize(cache, rest...);
}

template<typename Dtype, typename... Rest>
/**
 * [serialize description]
 * @param cache point of string
 * @param v     vector of Dtype data
 * @param rest  the rest of template data
 */
void serialize(std::string* cache, vector<Dtype>& v, Rest&... rest) {
    serialize<Dtype>(cache, v);
    serialize(cache, rest...);
}

template<typename T>
/**
 * [deserialize deserialize a string into a data]
 * @param cache reference of string
 * @param t     the reference of a data need to be deserialize
 */
void deserialize(const std::string& cache, T& t) {
    auto work_ptr = reinterpret_cast<T*>(const_cast<char*>(cache.c_str()));
    t = *work_ptr;
}

/**
 * deserilize and it will cut string buffer
 */
template<typename T>
void mutable_deserialize(std::string& cache, T& t) {
    auto work_ptr = reinterpret_cast<T*>(const_cast<char*>(cache.c_str()));
    t = *work_ptr;
    //cut the cache
    cache = std::move(cache.substr(sizeof(T)));
}

template<typename Dtype>
/**
 * [deserialize description]
 * @param cache reference of string
 * @param v     vector of data referece
 */
void deserialize(const std::string& cache, vector<Dtype>& v) {
    auto work_ptr = reinterpret_cast<Dtype*>(const_cast<char*>(cache.c_str()));
    for (int i = 0; i < v.size(); ++i)
    {
        v[i] = work_ptr[i];
    }
}

/**
 * deseriaze and cut string buffer
 */
template<typename Dtype>
void mutable_deserialize(std::string& cache, vector<Dtype>& v) {
    auto work_ptr = reinterpret_cast<Dtype*>(const_cast<char*>(cache.c_str()));
    for (int i = 0; i < v.size(); ++i)
    {
        v[i] = work_ptr[i];
    }
    //cut cache
    cache = std::move(cache.substr(v.size() * sizeof(Dtype)));
}

template<typename T, typename... Rest>
/**
 * [deserialize description]
 * @param cache reference of string
 * @param t     the reference of normal type
 * @param rest  the rest of template data refence
 */
void deserialize(const std::string& cache, T& t, Rest&... rest) {
    deserialize<T>(cache, t);
    auto rest_str = cache.substr(sizeof(T));
    deserialize(rest_str, rest...);
}

template<typename T, typename... Rest>
void mutable_deserialize(std::string& cache, T& t, Rest&... rest) {
    mutable_deserialize<T>(cache, t);
    mutable_deserialize(cache, rest...);
}

template<typename Dtype, typename... Rest>
/**
 * [deserialize description]
 * @param cache the refence of string
 * @param t     the refence of normal type
 * @param rest  the rest of template data reference
 */
void deserialize(const std::string& cache, vector<Dtype>& t, Rest&... rest) {
    deserialize<Dtype>(cache, t);
    auto rest_str = cache.substr(sizeof(Dtype) * t.size());
    deserialize(rest_str, rest...);
}

template<typename Dtype, typename... Rest>
void mutable_deserialize(std::string& cache, vector<Dtype>& t, Rest&... rest) {
    mutable_deserialize<Dtype>(cache, t);
    mutable_deserialize(cache, rest...);
}

/**
 * write string buffer to file
 */
inline void write_buffer_to_disk(const std::string& buffer, const char* filename) {
    std::ofstream ofile(filename, std::ios::binary);
    ofile.write(buffer.c_str(), sizeof(char) * buffer.size());
    ofile.close();
}

/**
 * read disk content to buffer
 */
inline std::string read_buffer_from_disk(const char* filename) {
    std::ifstream ifile;
    ifile.open(filename);
    std::string buffer((std::istreambuf_iterator<char>(ifile)),
        std::istreambuf_iterator<char>());
    ifile.close();
    return std::move(buffer);
}


} //wzp

#endif /*SERIALIZE_HPP_*/