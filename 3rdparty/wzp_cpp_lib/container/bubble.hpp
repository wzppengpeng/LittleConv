#ifndef BUBBLE_HPP_
#define BUBBLE_HPP_

#include <cassert>

#include <vector>
#include <iostream>
#include <fstream>

#include "my_string/string.hpp"

using std::vector;
using std::cout;
using std::endl;

constexpr int ShapeSize = 4;

namespace wzp {

template<typename Dtype>
class Bubble
{
public:
    Bubble() = default;

    //the constructor by four params
    explicit Bubble(const int num, const int channel,
     const int height, const int width);

    explicit Bubble(const vector<int>& shape);

    /*reshape functions*/
    void reshape(const int num, const int channel,
        const int height, const int width);

    //change tensor into a matrix
    void reshape(const int num);

    //reshape by vector
    void reshape(const vector<int>& shape);

    //some getters
    //get all data count size
    /**
     * get the count, just the size of m_data
     * @return sizeof total data
     */
    inline size_t count() const {
        return m_count;
    }

    //get data count by index
    inline size_t count(size_t start, size_t end) const {
        size_t count_ = 1;
        for(auto i = start; i < end; ++i) {
            count_ *= m_shape[i];
        }
        return count_;
    }

    inline size_t count(size_t start) const {
        return count(start, ShapeSize);
    }

    //get the each index shape
    inline int num() const { return m_shape[0]; }

    inline int channel() const { return m_shape[1]; }

    inline int height() const { return m_shape[2]; }

    inline int width() const { return m_shape[3]; }

    //compute the offset
    inline size_t offset(const int n, const int c = 0,
     const int h = 0, const int w = 0) const {
        assert(n < num() && c < channel() && h < height() && w < width());
        return ((n * channel() + c ) * height() + h) * width() + w;
    }

    //get the read data by index
    inline Dtype& at(const int n, const int c = 0, const int h = 0, const int w = 0) {
        return m_data[offset(n, c, h, w)];
    }

    inline const Dtype& at(const int n, const int c = 0, const int h = 0, const int w = 0) const {
        return m_data[offset(n, c, h, w)];
    }

    //print the bubble
    void print_bubble() const;

    /*********************************/
    /*             place                  */
    //some place function to place data into bubble
    //this is for the double dim matrix
    void place(const vector<vector<Dtype>>& data);

    //place a hole vector of data into m_data by move
    void place(vector<Dtype>&& data);

    //place the raw point of memory data
    void place(const Dtype* data, size_t len);

    /*******************************/
    /*some point get function*/
    //get the data point by num
    inline const Dtype* ptr(const int n) const {
        return &m_data[offset(n)];
    }

    //get the data by vector
    inline vector<Dtype> get(const int n) {
        return vector<Dtype>(ptr(n), ptr(n) + count(1));
    }

    /**
     * change the bubble into string(memory copy)
     * @return std::string a char array
     */
    std::vector<char> to_char_array() noexcept {
        std::vector<char> cache;
        auto total_size = ShapeSize * sizeof(int) + sizeof(Dtype) * m_count;
        cache.resize(total_size);
        //first store the num, channel, height, width
        auto num_ptr = reinterpret_cast<char*>(&m_shape[0]);
        size_t i;
        for(i = 0; i < ShapeSize * sizeof(int); ++i) {
            cache[i] = num_ptr[i];
        }
        auto data_ptr = reinterpret_cast<char*>(&m_data[0]);
        for(; i < total_size; ++i) {
            cache[i] = data_ptr[i - ShapeSize * sizeof(int)];
        }
        return std::move(cache);
    }

    /**
     * save this bubble into a binary file, save params first then save data
     * @param bin_file the binnary file name
     * @return void
     */
    inline void to_bin_file(const char* bin_file) noexcept {
        std::ofstream ofile(bin_file, std::ios::binary);
        auto cache = to_char_array();
        ofile.write(&cache[0], cache.size());
        ofile.close();
    }

    /**
     * read memory char array data into the m_data and m_shape
     * @param cache the memory char cache
     */
    void from_char_array(std::vector<char>& cache) noexcept {
        auto shape_ptr = reinterpret_cast<int*>(&cache[0]);
        std::vector<int> shape(shape_ptr, shape_ptr + ShapeSize);
        size_t cnt = 1;
        reshape(shape);
        auto data_ptr = reinterpret_cast<Dtype*>(&cache[ShapeSize * sizeof(int)]);
        for(size_t i = 0; i < count(); ++i) {
            m_data[i] = data_ptr[i];
        }
    }

    /**
     * read data from binary file, first read into the char arry, then call from_char_array
     * @param bin_file the binary file name
     */
    inline void from_bin_file(const char* bin_file) noexcept {
        std::ifstream ifile(bin_file, std::ios::binary);
        std::vector<char> cache(ShapeSize * sizeof(int));
        ifile.read(&cache[0], cache.size());
        auto shape_ptr = reinterpret_cast<int*>(&cache[0]);
        std::vector<int> shape(shape_ptr, shape_ptr + ShapeSize);
        size_t cnt = 1;
        for(auto s : shape) {
            cnt *= s;
        }
        auto total_size = cache.size() + sizeof(Dtype) * cnt;
        cache.resize(total_size);
        ifile.read(&cache[ShapeSize * sizeof(int)], sizeof(Dtype) * cnt);
        from_char_array(cache);
        ifile.close();
    }

    //decalare the friend
    friend std::ostream &operator<<(std::ostream &os,const Bubble<float> &b);
    friend std::ostream &operator<<(std::ostream &os, const Bubble<double> &b);

    friend std::istream &operator>>(std::istream &is, Bubble<float> &b);
    friend std::istream &operator>>(std::istream &is, Bubble<double> &b);
private:
    //reallocate memory to store data
    inline void reallocate() {
        m_data.reserve(m_capacity);
        m_data.resize(m_capacity);
    }


protected:
    //the data
    std::vector<Dtype> m_data;
    //the shape vector, size is 4
    std::vector<int> m_shape {0, 0, 0, 0};//num, channel, height, width
    //the real size of the hole data
    size_t m_count = 0;// now size all

    size_t m_capacity = 0;// the m_data's now size as m_data.capacity(), reserve

};


//implemente
template<typename Dtype>
void Bubble<Dtype>::reshape(const int num, const int channel,
        const int height, const int width) {
    vector<int> shape(ShapeSize);
    shape[0] = num;
    shape[1] = channel;
    shape[2] = height;
    shape[3] = width;
    reshape(shape);
}

template<typename Dtype>
void Bubble<Dtype>::reshape(const int num) {
    vector<int> shape(ShapeSize);
    shape[0] = num;
    shape[1] = channel() * height() * width();
    shape[2] = 1;
    shape[3] = 1;
    reshape(shape);
}

template<typename Dtype>
void Bubble<Dtype>::reshape(const vector<int>& shape) {
    assert(shape.size() == ShapeSize);
    m_count = 1;
    for(auto i = 0; i < ShapeSize; ++i) {
        m_count *= shape[i];
        m_shape[i] = shape[i];
    }
    if(m_count > m_capacity) {
        m_capacity = m_count;
        //reallcate data
        reallocate();
    }
}

template<typename Dtype>
Bubble<Dtype>::Bubble(const int num, const int channel,
     const int height, const int width) {
    reshape(num, channel, height, width);
}

template<typename Dtype>
Bubble<Dtype>::Bubble(const vector<int>& shape) {
    reshape(shape);
}

template<typename Dtype>
void Bubble<Dtype>::print_bubble() const {
    cout<<"[";
    for(int n = 0; n < num(); ++n) {
        cout<<"\n[";
        for(int c = 0; c < channel(); ++c) {
            cout<<" [";
            for(int h = 0; h < height(); ++h) {
                cout<<"[";
                for(int w = 0; w < width(); ++w) {
                    cout<<"[";
                    cout<<m_data[offset(n, c, h, w)]<<" ";
                    cout<<"]";
                }
                cout<<"],";
            }
            cout<<"] ";
        }
        cout<<"]\n";
    }
    cout<<"]"<<endl;
}

template<typename Dtype>
void Bubble<Dtype>::place(const vector<vector<Dtype>>& data) {
    assert(data.size() == num() && data[0].size() == channel() && height() == 1 && width() == 1);
    for(auto i = 0; i < data.size(); ++i) {
        for(auto j = 0; j < data[0].size(); ++j) {
            this->at(i, j) = data[i][j];
        }
    }
}

template<typename Dtype>
void Bubble<Dtype>::place(vector<Dtype>&& data) {
    assert(data.size() == count());
    m_data = data;//move the data into here
}

template<typename Dtype>
void Bubble<Dtype>::place(const Dtype* data, size_t len) {
    assert(len == count());
    vector<Dtype> data_vec(data, data + len);
    place(std::move(data_vec));
}

std::ostream &operator<<(std::ostream &os,const Bubble<float> &b) {
    os<<b.num()<<':'<<b.channel()<<':'<<b.height()<<':'<<b.width()<<'#';
    for(size_t i = 0; i < b.count() - 1; ++i) {
        os<<b.m_data[i]<<'|';
    }
    os<<b.m_data[b.count() - 1];
    return os;
}

std::ostream &operator<<(std::ostream &os, const Bubble<double> &b) {
    os<<b.num()<<':'<<b.channel()<<':'<<b.height()<<':'<<b.width()<<'#';
    for(size_t i = 0; i < b.count() - 1; ++i) {
        os<<b.m_data[i]<<'|';
    }
    os<<b.m_data[b.count() - 1];
    return os;
}

std::istream &operator>>(std::istream &is, Bubble<float> &b) {
    std::string cache;
    is >> cache;
    auto bubble_vec = wzp::split_string(cache, '#');
    auto param_vec = wzp::split_string(bubble_vec[0], ':');
    int n = std::stoi(param_vec[0]);
    int c = std::stoi(param_vec[1]);
    int h = std::stoi(param_vec[2]);
    int w = std::stoi(param_vec[3]);
    b.reshape(n, c, h, w);
    auto data_vec = wzp::split_string(bubble_vec[1], '|');
    for(size_t i = 0; i < b.count(); ++i) {
        b.m_data[i] = std::stof(data_vec[i]);
    }
    return is;
}

std::istream &operator>>(std::istream &is, Bubble<double> &b) {
    std::string cache;
    is >> cache;
    auto bubble_vec = wzp::split_string(cache, '#');
    auto param_vec = wzp::split_string(bubble_vec[0], ':');
    int n = std::stoi(param_vec[0]);
    int c = std::stoi(param_vec[1]);
    int h = std::stoi(param_vec[2]);
    int w = std::stoi(param_vec[3]);
    b.reshape(n, c, h, w);
    auto data_vec = wzp::split_string(bubble_vec[1], '|');
    for(size_t i = 0; i < b.count(); ++i) {
        b.m_data[i] = std::stod(data_vec[i]);
    }
    return is;
}

} // wzp

#endif /*BUBBLE_HPP_*/