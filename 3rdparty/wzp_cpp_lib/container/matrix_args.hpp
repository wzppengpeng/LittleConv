#ifndef MATRIX_ARGS_HPP_
#define MATRIX_ARGS_HPP_


#include "container/array_args.hpp"

namespace wzp
{

//some alias
template<typename T>
using Vec = std::vector<T>;

template<typename T>
using Matrix = std::vector<Vec<T> >;

/**
 * the vector<vector<T> > op args
 */
template<typename T>
class matrix_args
{

public:
    /**
     * get the max value's index
     */
    inline static Vec<size_t> arg_max_row(const Matrix<T>& mat) {
        Vec<size_t> row_index(mat.size());
        for(size_t i = 0; i < mat.size(); ++i) {
            row_index[i] = array_args<T>::arg_max(mat[i]);
        }
        return std::move(row_index);
    }

    /**
     * get the min values's index
     */
    static Vec<size_t> arg_max_col(const Matrix<T>& mat) {
        assert(mat.empty() == false && mat[0].empty() == false);
        Vec<size_t> col_index(mat[0].size());
        for(size_t j = 0; j < mat[0].size(); ++j) {
            size_t max_index = 0;
            for(size_t i = 1; i < mat.size(); ++i) {
                if(mat[i][j] > mat[max_index][j]) max_index = i;
            }
            col_index[j] = max_index;
        }
        return std::move(col_index);
    }

    inline static Vec<size_t> arg_min_row(const Matrix<T>& mat) {
        Vec<size_t> row_index(mat.size());
        for(size_t i = 0; i < mat.size(); ++i) {
            row_index[i] = array_args<T>::arg_min(mat[i]);
        }
        return std::move(row_index);
    }

    static Vec<size_t> arg_min_col(const Matrix<T>& mat) {
        assert(mat.empty() == false && mat[0].empty() == false);
        Vec<size_t> col_index(mat[0].size());
        for(size_t j = 0; j < mat[0].size(); ++j) {
            size_t max_index = 0;
            for(size_t i = 1; i < mat.size(); ++i) {
                if(mat[i][j] < mat[max_index][j]) max_index = i;
            }
            col_index[j] = max_index;
        }
        return std::move(col_index);
    }


    /**
     * sum functions
     */
    inline static Vec<T> sum_row(const Matrix<T>& mat) {
        Vec<T> row_sums(mat.size());
        for(size_t i = 0; i < mat.size(); ++i) {
            row_sums[i] = array_args<T>::sum(mat[i]);
        }
        return std::move(row_sums);
    }

    static Vec<T> sum_col(const Matrix<T>& mat) {
        assert(mat.empty() == false && mat[0].empty() == false);
        Vec<T> col_sums(mat[0].size());
        for(size_t j = 0; j < mat[0].size(); ++j) {
            T sum_val = 0;
            for(size_t i = 0; i < mat.size(); ++i) {
                sum_val += mat[i][j];
            }
            col_sums[j] = sum_val;
        }
        return std::move(col_sums);
    }

};


} // wzp


#endif /*MATRIX_ARGS_HPP_*/