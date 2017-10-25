#ifndef LICON_UTILS_SERILIALIZE_HPP
#define LICON_UTILS_SERILIALIZE_HPP

/**
 * serialize something into a string and deserialize something into type
 */
#include <cassert>
#include <string>
#include <vector>


// some specaial type declaration


namespace licon
{

namespace utils
{


// the tensor type
template<typename Dtype>
class ETensor;


// the static class of serialize
class Serializer {

public:
    // serialize functions
    // the single variable
    template<typename T>
    static void serialize(T& t, std::string* buffer) {
        std::string temp(sizeof(T), ' ');
        char* work_ptr = reinterpret_cast<char*>(&t);
        for(size_t i = 0; i < sizeof(T); ++i) {
            temp[i] = work_ptr[i];
        }
        buffer->append(std::move(temp));
    }

    // serialize the special string
    inline static void serialize(std::string t, std::string* buffer) {
        size_t len = t.size();
        serialize(len, buffer);
        buffer->append(std::move(t));
    }


    // serialize pointer memory data
    template<typename T, typename Index>
    static void serialize(T* t_ptr, Index len, std::string* buffer) {
        std::string temp(sizeof(T) * len, ' ');
        T* ptr = t_ptr;
        char* work_ptr = reinterpret_cast<char*>(ptr);
        for(size_t i = 0; i < temp.size(); ++i) {
            temp[i] = work_ptr[i];
        }
        buffer->append(std::move(temp));
    }

    // serialize the vector data(need to be the memory align data)
    template<typename T>
    inline static void serialize(std::vector<T>& v, std::string* buffer) {
        serialize(&v[0], v.size(), buffer);
    }

    // serialize the etensor
    template<typename Dtype>
    inline static void serialize(utils::ETensor<Dtype>& tensor, std::string* buffer) {
        // first serialize the shape
        auto shape = tensor.shape();
        serialize(shape, buffer);
        // serialize the data
        serialize(tensor.mutable_ptr(), tensor.count(), buffer);
    }

    // deserialize functions
    // desirialize the single object
    template<typename T>
    static size_t deserialize(std::string& buffer, size_t begin, T& t) {
        assert(begin < buffer.size());
        char* t_ptr = &buffer[begin];
        T* work_ptr = reinterpret_cast<T*>(t_ptr);
        t = *work_ptr;
        // return the new begin
        return begin + sizeof(T);
    }

    // deserialize the string
    inline static size_t deserialize(std::string& buffer, size_t begin, std::string& str) {
        assert(begin < buffer.size());
        size_t len;
        begin = deserialize(buffer, begin, len);
        char* loc = &buffer[begin];
        str = std::string(loc, loc + len);
        return begin + len;
    }

    // deserialize pointer number
    template<typename T, typename Index>
    static size_t deserialize(std::string& buffer, size_t begin, T* data_ptr, Index len) {
        assert(begin < buffer.size());
        char* ptr = &buffer[begin];
        T* work_ptr = reinterpret_cast<T*>(ptr);
        for(Index i = 0; i < len; ++i) {
            data_ptr[i] = work_ptr[i];
        }
        return begin + static_cast<size_t>(len) * sizeof(T);
    }

    // deserialize the vector
    template<typename T>
    inline static size_t deserialize(std::string& buffer, size_t begin, std::vector<T>& v) {
        return deserialize(buffer, begin, &v[0], v.size());
    }

    // deserialize the tensor
    template<typename Dtype>
    inline static size_t deserialize(std::string& buffer, size_t begin, utils::ETensor<Dtype>& tensor) {
        // first is the shape
        int num, channel, height, width;
        begin = deserialize(buffer, begin, num);
        begin = deserialize(buffer, begin, channel);
        begin = deserialize(buffer, begin, height);
        begin = deserialize(buffer, begin, width);
        tensor.Reshape(num, channel, height, width);
        return deserialize(buffer, begin, tensor.mutable_ptr(), tensor.count());
    }
};

} //utils

} //licon


#endif /*LICON_UTILS_SERILIALIZE_HPP*/