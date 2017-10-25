#ifndef TENSOR_HPP_
#define TENSOR_HPP_

#include <cassert>

#include <vector>
#include <utility>


namespace wzp
{

using std::vector;
using std::pair;


template<typename T>
class Tensor
{

private:
    //the data container
    vector<T> m_data;
    int m_height = 0;
    int m_width = 0;
    int m_channel = 0;

public:
    /**
     * Constructors
     */
    Tensor() = default;

    //create by sclar and fixed size
    Tensor(int h, int w, int c, int init_val = 0) : m_data(h * w * c, init_val),
                                                    m_height(h),
                                                    m_width(w),
                                                    m_channel(c)
                                                    {}


    //move constructor
    Tensor(Tensor<T>&& other) noexcept : m_data(std::move(other.m_data)),
                                m_height(other.m_height),
                                m_width(other.m_width),
                                m_channel(other.m_channel)
    {}

    Tensor<T>& operator= (Tensor<T>&& other) noexcept {
        assert(this != &other);
        m_data = std::move(other.m_data);
        m_height = other.m_height;
        m_width = other.m_width;
        m_channel = other.m_channel;
        return *this;
    }


    //copy constructor
    Tensor(const Tensor<T>&) = default;
    Tensor<T>& operator= (const Tensor<T>&) = default;


    //reshape
    void reshape(int new_h, int new_w, int new_c) {
        m_height = new_h;
        m_width = new_w;
        m_channel = new_c;
        int new_capacity = new_h * new_h * new_c;
        if(new_capacity > m_data.size()) {
            m_data.resize(new_capacity);
        }
    }

    //some getter
    inline bool empty() const { return m_height == 0; }
    inline int height() const { return m_height; }
    inline int width() const { return m_width; }
    inline int channel() const { return m_channel; }

    //location data getter
    inline T at(int h, int w, int c) const {
        return m_data[idx(h, w, c)];
    }

    inline T& at(int h, int w, int c) {
        return m_data[idx(h, w, c)];
    }

    inline T operator() (int h, int w, int c) const {
        return at(h, w, c);
    }

    inline T& operator() (int h, int w, int c) {
        return at(h, w, c);
    }

    inline vector<T> get_channel_at(int h, int w) const {
        return vector<T>(m_data.begin() + idx(h, w, 0), m_data.begin() + idx(h, w, m_channel));
    }

    inline const T* channel_ptr(int h, int w) const {
        return &m_data[idx(h, w, 0)];
    }

    //this is unsafe
    inline T* channel_ptr(int h, int w) {
        return &m_data[idx(h, w, 0)];
    }


    // Slice Function
    Tensor<T> slice(pair<int, int> height_range,
        pair<int, int> width_range, pair<int, int> channel_range) {
        assert(height_range.second <= m_height && width_range.second <= m_width && channel_range.second <= m_channel);
        //wash the range
        if(height_range.second < 0) height_range.second = height();
        if(width_range.second < 0) width_range.second = width();
        if(channel_range.second < 0) channel_range.second = channel();
        Tensor<T> res(height_range.second - height_range.first,
            width_range.second - width_range.first, channel_range.second - channel_range.first);
        for(int h = height_range.first; h < height_range.second; ++h) {
            for(int w = width_range.first; w < width_range.second; ++w) {
                for(int c = channel_range.first; c < channel_range.second; ++c) {
                    res(h - height_range.first, w - width_range.first,
                        c - channel_range.first) = at(h, w, c);
                }
            }
        }
        return std::move(res);
    }

    //slice for height and width
    Tensor<T> slice(const pair<int, int>& height_range, const pair<int, int>& width_range) {
        return std::move(slice(height_range, width_range, std::make_pair(0, -1)));
    }


private:
    //the index function
    inline int idx(int h, int w, int c) const {
        assert(h < height() && w < width() && c < channel());
        return (h * width() + w) * channel() + c;
    }


};


} //wzp


#endif //TENSOR_HPP_