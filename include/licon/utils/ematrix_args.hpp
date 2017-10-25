#ifndef LICON_E_MATRIX_ARGS_HPP_
#define LICON_E_MATRIX_ARGS_HPP_

#include "container/ematrix.hpp"
#include "licon/utils/math.hpp"

namespace licon
{

namespace utils
{

template<typename Dtype>
class EMatrixArgs {
public:
    /**
     * fill the matrix with some values
     */
    // fill the matrix with a scalar
    inline static void fill(wzp::EMatrix<Dtype>& matrix, Dtype val) {
        for(int i = 0; i < matrix.rows(); ++i) {
            for(int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = val;
            }
        }
    }

    inline static void normal(wzp::EMatrix<Dtype>& matrix, Dtype mean, Dtype var) {
        for(int i = 0; i < matrix.rows(); ++i) {
            for(int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = utils::licon_random_normal(mean, var);
            }
        }
    }

    inline static void uniform(wzp::EMatrix<Dtype>& matrix, Dtype low, Dtype high) {
        for(int i = 0; i < matrix.rows(); ++i) {
            for(int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = utils::licon_random_real(low, high);
            }
        }
    }

    // copy the matrix value to pointer one by one
    inline static void copy_to_pointer(const wzp::EMatrix<Dtype>& matrix, Dtype* ptr) {
        int cnt = matrix.rows() * matrix.cols();
        for(int i = 0; i < cnt; ++i) {
            ptr[i] = matrix(i);
        }
    }

    // add the matrix value to pointer
    inline static void add_to_pointer(const wzp::EMatrix<Dtype>& matrix, Dtype* ptr) {
        int cnt = matrix.rows() * matrix.cols();
        for(int i = 0; i < cnt; ++i) {
            ptr[i] += matrix(i);
        }
    }

};

} //utils

} //licon


#endif /*LICON_E_MATRIX_ARGS_HPP_*/