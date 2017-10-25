/**
 * the transformer of tensor to matrix or matrix to tensor
 */

#include "licon/utils/etensor.hpp"
#include "container/ematrix.hpp"

namespace licon
{

namespace utils
{

// the transform from etensor to ematrix
template<typename Dtype>
void transform_tensor_to_matrix(const ETensor<Dtype>& t, wzp::EMatrix<Dtype>& m) {
    assert(t.num() == m.rows() && t.count(1) == m.cols());
    for(int i = 0; i < t.num(); ++i) {
        const Dtype* d_ptr = t.ptr(i);
        for(int j = 0; j < t.count(1); ++j) {
            m(i, j) = d_ptr[j];
        }
    }
}

// transform from ematrix to ettensor
template<typename Dtype>
void transform_matrix_to_tensor(const wzp::EMatrix<Dtype>& m, utils::ETensor<Dtype>& t) {
    assert(m.rows() == t.num() && m.cols() == t.count(1));
    for(int i = 0; i < m.rows(); ++i) {
        Dtype* d_ptr = t.mutable_ptr(i);
        for(int j = 0; j < m.cols(); ++j) {
            d_ptr[j] = m(i, j);
        }
    }
}

// transform the vector<T> to etensor
template<typename T, typename U>
void transform_vector_to_tensor(utils::ETensor<T>& t, const std::vector<U>& v) {
    // will put data one the first dim of tensor
    t.Reshape(v.size(), 1, 1, 1);
    T* t_ptr = t.mutable_ptr();
    for(size_t i = 0; i < v.size(); ++i) {
        t_ptr[i] = static_cast<T>(v[i]);
    }
}

// transform the etensor to vector<T>
template<typename T, typename U>
void transform_tensor_to_vector(const utils::ETensor<T>& t, std::vector<U>& v) {
    v.clear();
    v.reserve(t.count());
    T* t_ptr = t.ptr();
    for(size_t i = 0; i < t.count(); ++i) {
        v[i] = static_cast<U>(t_ptr[i]);
    }
}

} //utils

} //licon