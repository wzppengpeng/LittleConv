#ifndef LICON_E_TENSOR_HPP
#define LICON_E_TENSOR_HPP

#include <cassert>

#include <vector>
#include <iostream>

constexpr int ShapeSize = 4;

namespace licon
{

namespace utils
{

template<typename Dtype>
class ETensor {

public:
    // the default constructor
    ETensor() = default;

    explicit ETensor(const std::vector<int>& shape) { Reshape(shape); }

    explicit ETensor(const int num, const int channel, const int height, const int width) { Reshape(num, channel, height, width); }

    ETensor(const ETensor<Dtype>& other) = default;
    ETensor<Dtype>& operator= (const ETensor<Dtype>& other) = default;

    ETensor(ETensor<Dtype>&& other) = default;
    ETensor<Dtype>& operator= (ETensor<Dtype>&& other) = default;

    // the reshape function
    void Reshape(const int num, const int channel,
        const int height, const int width) {
        std::vector<int> shape(ShapeSize);
        shape[0] = num;
        shape[1] = channel;
        shape[2] = height;
        shape[3] = width;
        Reshape(shape);
    }

    void Reshape(const int num) {
        std::vector<int> shape(ShapeSize);
        shape[0] = num;
        shape[1] = channel() * height() * width();
        shape[2] = 1;
        shape[3] = 1;
        Reshape(shape);
    }

    void Reshape(const std::vector<int>& shape) {
        assert(shape.size() == ShapeSize);
        m_count = 1;
        for(int i = 0; i < ShapeSize; ++i) {
            m_count *= shape[i];
            m_shape[i] = shape[i];
        }
        if(m_count > m_capacity) {
            m_capacity = m_count;
            reallocate(); //reallcate data
        }
    }

    // the view function, to set the tensor into a 2-d matrix
    template<typename Mat>
    Mat View() const {
        int col = count(1);
        Mat mat(num(), col);
        for(int n = 0; n < num(); ++n) {
            for(int j = 0; j < col; ++j) {
                mat(n, j) = m_data[n * col + j];
            }
        }
        return std::move(mat);
    }


    // the view function inplace
    template<typename Mat>
    Mat ViewInPlace() {
        int col = count(1);
        return std::move(Mat(num(), col, mutable_ptr()));
    }


    // getters
    // count the size of total data
    inline size_t count() const { return m_count; }

    //get data count by index
    inline size_t count(size_t start, size_t end) const {
        size_t count_ = 1;
        for(auto i = start; i < end; ++i) {
            count_ *= m_shape[i];
        }
        return count_;
    }

    // the count to end
    inline size_t count(size_t start) const { return count(start, ShapeSize); }

    //get the each index shape
    inline int num() const { return m_shape[0]; }
    inline int channel() const { return m_shape[1]; }
    inline int height() const { return m_shape[2]; }
    inline int width() const { return m_shape[3]; }
    inline std::vector<int> shape() const { return m_shape; }

    //compute the offset
    inline size_t offset(const int n, const int c = 0,
     const int h = 0, const int w = 0) const {
        assert(n < num() && c < channel() && h < height() && w < width());
        return ((n * channel() + c ) * height() + h) * width() + w;
    }

    //get the read data by index
    inline Dtype& at(const int n, const int c = 0, const int h = 0, const int w = 0) { return m_data[offset(n, c, h, w)]; }
    inline Dtype at(const int n, const int c = 0, const int h = 0, const int w = 0) const { return m_data[offset(n, c, h, w)]; }
    inline Dtype& operator() (const int n, const int c = 0, const int h = 0, const int w = 0) { return at(n, c, h, w); }
    inline Dtype operator() (const int n, const int c = 0, const int h = 0, const int w = 0) const { return at(n, c, h ,w); }

    // place data into etensor
    //place a hole vector of data into m_data by move
    void Place(std::vector<Dtype>&& data) {
        assert(data.size() == count());
        m_data = std::move(data);
    }

    //Place the raw point of memory data
    void Place(const Dtype* data, size_t len) {
        assert(len == count());
        std::vector<Dtype> data_vec(data, data + len);
        Place(std::move(data_vec));
    }

    /*some point get function*/
    //get the data point by num
    inline const Dtype* ptr(const int n = 0) const { return &m_data[offset(n)]; }
    // get the mutable ptr
    inline Dtype* mutable_ptr(int n = 0) { return &m_data[offset(n)]; }
    //get the data by vector
    inline std::vector<Dtype> get(const int n) const { return std::vector<Dtype>(ptr(n), ptr(n) + count(1)); }

    /**
     * the print functions into ostream
     */
    inline friend std::ostream& operator << (std::ostream& os, const ETensor<Dtype>& tensor) {
        os << "[";
        for(int n = 0; n < tensor.num(); ++n) {
            os << "\n[";
            for(int c = 0; c < tensor.channel(); ++c) {
                os << " [";
                for(int h = 0; h < tensor.height(); ++h) {
                    os << "[";
                    for(int w = 0; w < tensor.width(); ++w) {
                        os << tensor.m_data[tensor.offset(n, c, h, w)] << ' ';
                        os << ",";
                    }
                    os << "],";
                }
                os << "] ";
            }
            os << "]\n";
        }
        os << "]"<< std::endl;
        return os;
    }

protected:
    /**
     * the member data
     */
    // the data container
    std::vector<Dtype> m_data;
    //the shape vector, size is 4
    std::vector<int> m_shape {0, 0, 0, 0};//num, channel, height, width
    //the real size of the hole data
    size_t m_count = 0;// now size all

    size_t m_capacity = 0;// the m_data's now size as m_data.capacity(), reserve

private:
    // private functions
    //reallocate memory to store data
    inline void reallocate() { m_data.resize(m_capacity); }

};

} //utils

} //licon


#endif /*LICON_E_TENSOR_HPP*/