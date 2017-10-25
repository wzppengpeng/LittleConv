#ifndef WZP_MATRIX_HPP_
#define WZP_MATRIX_HPP_

#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <iterator>

#include "thread/taskjob.hpp"
#include "my_string/string.hpp"
#include "util/serialize.hpp"

using std::vector;


namespace wzp
{

namespace details
{
/**
 * Operators!!!!!
 */
/**
 * DotProduct, N is the dim, T is the type
 */
template<size_t N, typename T>
class DotProduct {
public:
    static T eval(T* a, T* b) {
        return DotProduct<1, T>::eval(a, b) + DotProduct<N - 1, T>::eval(a + 1, b + 1);
    }
};

/**
 * Special
 */
template<typename T>
class DotProduct<1, T> {
public:
    static T eval(T* a, T* b) {
        return (*a) * (*b);
    }
};

template<typename T, size_t N>
struct MatDot
{
    static T eval(T* a, T* b, size_t C) {
        return MatDot<T, 1>::eval(a, b, C) + MatDot<T, N-1>::eval(a + 1, b + C, C);
    }
};

template<typename T>
struct MatDot<T, 1>
{
    static T eval(T* a, T* b, size_t C) {
        return (*a) * (*b);
    }
};

} //details

//the dot function
template<size_t N, typename T>
inline T dot(T*a, T* b) {
    return details::DotProduct<N, T>::eval(a, b);
}

template<typename T, size_t N>
inline T _dot(T* a, T* b, size_t C) {
    return details::MatDot<T, N>::eval(a, b, C);
}

namespace linear {

/**
 * Det of Matrix
 */
template<typename T=double, typename Mat>
T det(const Mat& arcs) {
    assert(arcs.rows() == arcs.cols());
    size_t n = arcs.rows();
    if(n == 1) return arcs(0, 0);
    T ans = 0.0;
    Mat temp(n - 1, n - 1);
    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n - 1; ++j) {
            for(size_t k = 0; k < n - 1; ++k) {
                temp(j, k) = arcs(j + 1, (k >= i ? k + 1 : k));
            }
        }
        auto t = det<T>(temp);
        if(i % 2 == 0) {
            ans += arcs(0, i) * t;
        }
        else {
            ans -= arcs(0, i) * t;
        }
    }
    return ans;
}
}//linear

/**
 * class of Matrix!!
 */

/**
 * Matrix Base, provite operators to avoid code bloat
 */
template<typename T>
class MatrixBase {};

/**
 * the Matrix!
 * M is the matrix rows
 * N is the matrix cols
 */

/**
 * Matrix Init Types
 */
enum class MatrixType {
    Zeros,
    Eyes
};
/**
 * Process Type
 */
enum class ProcessType {
    SingleThread,
    MultiThread
};

template<typename T>
class Matrix : private MatrixBase<T> {

public:
    vector<T> m_data {}; // the data container
    size_t m_row = 0;
    size_t m_col = 0; // Row and Col;
    ProcessType m_type = ProcessType::MultiThread;

public:
    /**
     * Constructors
     */
    Matrix() = default;

    //by a scalar value
    Matrix(size_t M, size_t N, T init_val = 0) : m_data(M * N, init_val)
                                            , m_row(M)
                                            , m_col(N)
    {}

    //handle the type of eye
    Matrix(size_t M, MatrixType type, T init_val = 1) : m_data(M * M, 0)
                                            , m_row(M)
                                            , m_col(M)
    {
        if(type == MatrixType::Eyes) {
            init(init_val);
        }
    }

    //init by vector<T> this is only for inner construct data
    Matrix(size_t M, size_t N, vector<T>&& data) : m_data(data)
                                            , m_row(M)
                                            , m_col(N)
    {}

    //init by vector<vector<T>>
    Matrix(size_t M, size_t N, const vector<vector<T>>& data) : m_data(M * N)
                                            , m_row(M)
                                            , m_col(N)
    {
        assert(M == data.size() && N == data[0].size());
        for(size_t i = 0; i < M; ++i) {
            for(size_t j = 0; j < N; ++j) {
                m_data[index(i, j)] = data[i][j];
            }
        }
    }

    //move constructor
    Matrix(Matrix<T>&& other) noexcept : m_data(std::move(other.m_data)),
                                m_row(other.m_row),
                                m_col(other.m_col),
                                m_type(other.m_type)
    {}

    Matrix<T>& operator= (Matrix&& other) noexcept {
        assert(this != &other);
        m_data = std::move(other.m_data);
        m_row = other.m_row;
        m_col = other.m_col;
        m_type = other.m_type;
        return *this;
    }

    //copy constructor
    Matrix(const Matrix<T>&) = default;
    Matrix<T>& operator= (const Matrix<T>&) = default;

    /**
     * [reshape description]
     * @param new_row
     * @param new_col
     */
    void reshape(size_t new_row, size_t new_col) {
        m_row = new_row;
        m_col = new_col;
        auto new_capacity = new_row * new_col;
        if(new_capacity > m_data.size()) {
            m_data.resize(new_capacity);
        }
    }

    /**
     * reverse
     */
    Matrix<T>& t() {
        vector<T> tmp(m_data);
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                tmp[j * m_row + i] = m_data[index(i, j)];
            }
        }
        std::swap(tmp, m_data);
        std::swap(m_row, m_col);
        return *this;
    }

    /**
     * reverse function const
     */
    Matrix<T> t_() const {
        vector<T> tmp(m_data);
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                tmp[j * m_row + i] = m_data[index(i, j)];
            }
        }
        return std::move(Matrix<T>(m_col, m_row, std::move(tmp)));
    }

    /**
     * Operations
     */
    //scalar product
    Matrix<T> operator*(T val) const {
        Matrix<T> res(*this);
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                res.at(i, j) *= val;
            }
        }
        return std::move(res);
    }

    Matrix<T>& operator*=(T val) {
        for(size_t i = 0; i < m_row * m_col; ++i) {
            m_data[i] *= val;
        }
        return *this;
    }

    //add function
    Matrix<T> operator+(const Matrix<T>& other) const {
        assert(m_row == other.rows() && m_col == other.cols());
        Matrix<T> res(m_row, m_col);
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                res(i, j) = this->at(i, j) + other(i, j);
            }
        }
        return std::move(res);
    }

    Matrix<T>& operator+=(const Matrix<T>& other) {
        assert(m_row == other.rows() && m_col == other.cols());
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                this->at(i, j) += other(i, j);
            }
        }
        return *this;
    }

    //minus fuunction
    Matrix<T> operator-(const Matrix<T>& other) const{
        assert(m_row == other.rows() && m_col == other.cols());
        Matrix<T> res(m_row, m_col);
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                res(i, j) = this->at(i, j) - other(i, j);
            }
        }
        return std::move(res);
    }

    Matrix<T>& operator-=(const Matrix<T>& other) {
        assert(m_row == other.rows() && m_col == other.cols());
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                this->at(i, j) -= other(i, j);
            }
        }
        return *this;
    }

    /**
     * Dot Product, C is the new Col
     */
    Matrix<T> operator*(const Matrix<T>& other) const {
        assert(m_col == other.rows());
        Matrix<T> res(m_row, other.cols());
        // check(other.cols());
        if(m_col > 256 || other.cols() > 256) {
            for(size_t i = 0; i < m_row; ++i) {
                for(size_t k = 0; k < other.cols(); ++k) {
                    T sum(0);
                    for(size_t j = 0; j < m_col; ++j) {
                        sum += (this->at(i, j) * other(j, k));
                    }
                    res(i, k) = sum;
                }
            }
        }
        else {
            for(size_t i = 0; i < m_row; ++i) {
                vector<size_t> cols_index;
                cols_index.reserve(other.cols());
                for(size_t k = 0; k < other.cols(); ++k) {
                    cols_index.push_back(k);
                }
                ParallelForeach(cols_index.begin(),
                 cols_index.end(), [&res, this, i, &other](size_t k){
                    T sum(0);
                    for(size_t j = 0; j < m_col; ++j) {
                        sum += at(i, j) * other(j, k);
                    }
                    res(i, k) = sum;
                });
            }
        }
        return res;
    }

    /**
     * Operations
     */
    Matrix<T> start() {
        Matrix<T> ans(m_row, m_row);
        if(m_row == 1) {
            ans(0, 0) = 1;
            return ans;
        }
        if(m_row <= 10) {
            Matrix<T> temp(m_row - 1, m_row - 1);
            for(size_t i = 0; i < m_row; ++i) {
                for(size_t j = 0; j < m_row; ++j) {
                    for(size_t k = 0; k < m_row - 1; ++k) {
                        for(size_t t = 0; t < m_row - 1; ++t) {
                            temp(k, t) = at((k >= i ? k + 1 : k), (t >= j ? t + 1 : t));
                        }
                    }
                    ans(j, i) = linear::det<T>(temp);
                    if((i + j) % 2 == 1) {
                        ans(j, i) = -ans(j, i);
                    }
                }
            }
        }
        else {
            vector<size_t> index(m_row);
            for(size_t i = 0; i < m_row; ++i) {
                index[i] = i;
            }
            ParallelForeach(index.begin(), index.end(),
             [&ans, this](size_t i) {
                Matrix<T> temp(m_row - 1, m_row - 1);
                for(size_t j = 0; j < m_row; ++j) {
                    for(size_t k = 0; k < m_row - 1; ++k) {
                        for(size_t t = 0; t < m_row - 1; ++t) {
                            temp(k, t) = at((k >= i ? k + 1 : k), (t >= j ? t + 1 : t));
                        }
                    }
                    ans(j, i) = linear::det<T>(temp);
                    if((i + j) % 2 == 1) {
                        ans(j, i) = -ans(j, i);
                    }
                }
            });
        }
        return ans;
    }

    Matrix<T> inv() {
        assert(m_row == m_col);
        auto flag = linear::det<T>(*this);
        assert(flag != 0);
        auto start_mat = this->start();
        start_mat *= static_cast<T>((1.0 / flag));
        return start_mat;
    }

    /**
     * Some getter
     */
    inline bool empty() const { return m_row == 0; }

    inline size_t rows() const { return m_row; }

    inline size_t cols() const { return m_col; }

    inline T at(size_t i, size_t j) const { return m_data[index(i, j)]; }
    inline T& at(size_t i, size_t j) { return m_data[index(i, j)]; }

    inline T operator() (size_t i, size_t j) const {
        return m_data[index(i, j)];
    }

    inline T& operator() (size_t i, size_t j) {
        return m_data[index(i, j)];
    }

    inline const T* row_at(size_t i ) const {
        return &m_data[index(i, 0)];
    }

    // inline T* row_at(size_t i) {
    //     return &m_data[index(i, 0)];
    // }

    /**
     * Set funtions
     */
    void set_process_type(ProcessType type) {
        m_type = type;
    }

    /**
     * print function
     */
    void print() const {
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                std::cout<<at(i, j)<<" ";
            }
            std::cout<<std::endl;
        }
    }

    /**
     * hard io function
     */
    /**
     * read from csv file
     * @param filename
     */
    void read_csv(const char* filename) noexcept {
        vector<T>().swap(m_data);
        std::ifstream ifile(filename, std::ios::in);
        std::string line;
        //read first line
        std::getline(ifile, line);
        auto strs = split_string(line, ',');
        m_col = strs.size();
        push_data(strs);
        size_t row_cnt = 1;
        //read other lines
        while(std::getline(ifile, line)) {
            strs = std::move(split_string(line, ','));
            push_data(strs);
            ++row_cnt;
        }
        m_row = row_cnt;
        ifile.close();
    }

    /**
     * trainsform to csv file
     * @param filename
     */
    void to_csv(const char* filename) const noexcept {
        if(m_data.empty()) return;
        std::ofstream ofile(filename, std::ios::out);
        for(size_t i = 0; i < m_row; ++i) {
            size_t j = 0;
            ofile<<at(i, j);
            for(j = 1; j < m_col; ++j) {
                ofile<<','<<at(i, j);
            }
            ofile<<std::endl;
        }
        ofile.close();
    }

    /**
     * trainsform to binary file
     * @param filename
     */
    inline void matrix_to_string(std::string* cache) {
        serialize(cache, m_row, m_col, m_data);
    }

    void to_bin_file(const char* filename) noexcept {
        if(m_data.empty()) return;
        std::ofstream ofile(filename, std::ios::binary);
        //serilize data into a string
        std::string cache;
        //first write the row and col nnumber, second the vector data
        serialize(&cache, m_row, m_col, m_data);
        ofile.write(cache.c_str(), sizeof(char) * cache.size());
        ofile.close();
    }

    inline void string_to_matrix(std::string& cache) {
        deserialize(cache, m_row, m_col);
        reshape(m_row, m_col);
        cache = std::move(cache.substr(2 * sizeof(decltype(m_row))));
        deserialize(cache, m_data);
    }

    void read_bin_file(const char* filename) noexcept {
        vector<T>().swap(m_data);
        std::ifstream ifile(filename);
        std::string file((std::istreambuf_iterator<char>(ifile)),
            std::istreambuf_iterator<char>());
        string_to_matrix(file);
    }

    /**
     * Slice
     */
    Matrix<T> slice(size_t row_begin, size_t col_begin,
     size_t row_end, size_t col_end) const {
        assert(row_end >= row_begin && col_end >= col_begin);
        assert(col_end <= cols());
        assert(row_end <= rows());
        Matrix<T> res(row_end - row_begin, col_end - col_begin);
        for(size_t i = row_begin; i < row_end; ++i) {
            for(size_t j = col_begin; j < col_end; ++j) {
                res(i - row_begin, j - col_begin) = at(i, j);
            }
        }
        return std::move(res);
    }

    /**
     * Filter, Set Matrix Values If A Condition Is Reached
     * the function should return bool, and input i, j
     */
    template<typename Function>
    void filter(const Function& fun, T val) {
        for(size_t i = 0; i < m_row; ++i) {
            for(size_t j = 0; j < m_col; ++j) {
                if(fun(i, j)) {
                    this->at(i, j) = val;
                }
            }
        }
    }

    /**
     * place one row data into matrix
     */
    void place(const T* data, size_t row_index) {
        assert(row_index < m_row);
        for(size_t j = 0; j < m_col; ++j) {
            at(row_index, j) = data[j];
        }
    }

private:
    void init(T init_val) {
        for(size_t i = 0; i < m_row; ++i) {
            m_data[index(i, i)] = init_val;
        }
    }
    //compute the index
    /**
     * compute the index
     * @param  i      x loc
     * @param  size_t y loc
     * @return        the index in m_data
     */
    inline size_t index(size_t i, size_t j) const {
        return i * m_col + j;
    }

    inline void check(size_t C) {
        if(m_col > 256 || C > 256) {
            m_type = ProcessType::MultiThread;
        }
        else {
            m_type = ProcessType::SingleThread;
        }
    }

    void push_data(const vector<std::string>& strs) {
        for(auto& str : strs) {
            m_data.emplace_back(convert_string<T>(str));
        }
    }

};

/**
 * Simple Special
 */




} //wzp

#endif /*WZP_MATRIX_HPP_*/