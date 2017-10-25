#ifndef WZP_EIGEN_MATRIX_HPP_
#define WZP_EIGEN_MATRIX_HPP_

/**
 * This is just a wapper for eigen matrix(DYNAMIC)
 */
#include <cassert>

#include <Eigen/Dense>


namespace wzp
{

template<typename Dtype>
class EMatrix {
private:
    // the alias for inner use of raw eigen matrix
    template<typename F>
    using RawMatrix = Eigen::Matrix<F, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // define the index type
    typedef Eigen::Index Index;

public:
    /**
     * Constructors
     */
    // default contructor
    EMatrix() : m_mat() {}

    // construct by size
    EMatrix(Index m) : m_mat(m, m) { InitByValue(0); }

    // construct by two size
    EMatrix(Index m, Index n, Dtype init_val = 0) : m_mat(m, n) {InitByValue(init_val); }

    // construct by raw point
    EMatrix(Index m, Index n, Dtype* raw_data_ptr) {
        m_mat = std::move(Eigen::Map<RawMatrix<Dtype>>(raw_data_ptr, m, n));
    }

    // construct by datas
    EMatrix(RawMatrix<Dtype>&& raw_mat) : m_mat(std::move(raw_mat)) {}
    EMatrix(const RawMatrix<Dtype>& raw_mat) : m_mat(raw_mat) {}

    // move construct
    EMatrix(EMatrix<Dtype>&& other) : m_mat(std::move(other.m_mat)) { }

    // move operator
    EMatrix<Dtype>& operator= (EMatrix<Dtype>&& other) {
        assert(this != &other);
        m_mat = std::move(other.m_mat);
        return *this;
    }

    // copy constructor
    EMatrix(const EMatrix<Dtype>&) = default;
    EMatrix<Dtype>& operator= (const EMatrix<Dtype>&) = default;

    // get the raw matrix
    inline RawMatrix<Dtype>& get_raw_mat() { return m_mat; }
    inline const RawMatrix<Dtype>& get_raw_mat() const { return m_mat; }

    /**
     * reshape functions
     */
    inline void reshape(Index m, Index n) {
        if(m == m_mat.rows() && n == m_mat.cols()) return;
        m_mat.resize(m, n);
    }

    /**
     * transpose and adjoint
     */
    // transpose
    EMatrix<Dtype> transpose() const {
        auto tmp = m_mat.transpose();
        return EMatrix<Dtype>(std::move(tmp));
    }

    // adjoint
    EMatrix<Dtype> adjoint() const {
        auto tmp = m_mat.adjoint();
        return EMatrix<Dtype>(std::move(tmp));
    }

    /**
     * operations by contant value
     */
    // scalar product
    EMatrix<Dtype> operator* (Dtype val) const {
        auto tmp = m_mat * val;
        return EMatrix<Dtype>(std::move(tmp));
    }

    EMatrix<Dtype>& operator*= (Dtype val) {
        m_mat *= val;
        return *this;
    }

    // scalar +
    EMatrix<Dtype> operator+ (Dtype val) const {
        auto tmp = m_mat + val;
        return EMatrix<Dtype>(std::move(tmp));
    }

    EMatrix<Dtype>& operator+= (Dtype val) {
        m_mat += val;
        return *this;
    }

    // scalar -
    EMatrix<Dtype> operator- (Dtype val) const {
        auto tmp = m_mat - val;
        return EMatrix<Dtype>(std::move(tmp));
    }

    EMatrix<Dtype>& operator-= (Dtype val) {
        m_mat -= val;
        return *this;
    }

    // matrix dot product
    EMatrix<Dtype> operator* (const EMatrix<Dtype>& other) const {
        auto tmp = m_mat * other.get_raw_mat();
        return EMatrix<Dtype>(std::move(tmp));
    }

    // matrix add function
    EMatrix<Dtype> operator+ (const EMatrix<Dtype>& other) const {
        if(other.rows() == 1) {
            assert(other.cols() == cols());
            EMatrix<Dtype> tmp_mat(*this);
            for(Index i = 0; i < rows(); ++i) {
                for(Index j = 0; j < cols(); ++j) {
                    tmp_mat.at(i, j) += other(0, j);
                }
            }
            return tmp_mat;
        } else if(other.cols() == 1) {
            assert(other.rows() == rows());
            EMatrix<Dtype> tmp_mat(*this);
            for(Index j = 0; j < cols(); ++j) {
                for(Index i = 0; i < rows(); ++i) {
                    tmp_mat.at(i, j) += other(i, 0);
                }
            }
            return tmp_mat;
        } else {
            auto tmp = m_mat + other.get_raw_mat();
            return EMatrix<Dtype>(std::move(tmp));
        }
    }

    EMatrix<Dtype>& operator+= (const EMatrix<Dtype>& other) {
        if(other.rows() == 1) {
            assert(other.cols() == cols());
            for(Index i = 0; i < rows(); ++i) {
                for(Index j = 0; j < cols(); ++j) {
                    at(i, j) += other(0, j);
                }
            }
        } else if(other.cols() == 1) {
            assert(other.rows() == rows());
            EMatrix<Dtype> tmp_mat(*this);
            for(Index j = 0; j < cols(); ++j) {
                for(Index i = 0; i < rows(); ++i) {
                    at(i, j) += other(i, 0);
                }
            }
        } else {
            m_mat += other.get_raw_mat();
        }
        return *this;
    }

    /**
     * Block
     */
    EMatrix<Dtype> block(Index i, Index j, Index h, Index w) const {
        auto tmp = m_mat.block(i, j, h, w);
        return EMatrix<Dtype>(std::move(tmp));
    }

    EMatrix<Dtype> row(Index i) const {
        auto tmp = m_mat.row(i);
        return EMatrix<Dtype>(std::move(tmp));
    }

    EMatrix<Dtype> col(Index j) const {
        auto tmp = m_mat.col(j);
        return EMatrix<Dtype>(std::move(tmp));
    }

    void mutable_row(Index i, const EMatrix<Dtype>& row_vec) {
        m_mat.row(i) = row_vec.get_raw_mat();
    }

    void mutable_col(Index j, const EMatrix<Dtype>& col_vec) {
        m_mat.col(j) = col_vec.get_raw_mat();
    }

    /**
     * Getters
     */
    inline bool empty() const { return m_mat.rows() == 0; }
    inline Index rows() const { return m_mat.rows(); }
    inline Index cols() const { return m_mat.cols(); }
    inline Dtype at(Index i, Index j) const { return m_mat(i, j); }
    inline Dtype& at(Index i, Index j) { return m_mat(i, j); }
    inline Dtype operator() (Index i, Index j) const { return at(i, j); }
    inline Dtype& operator() (Index i, Index j) { return at(i, j); }
    inline Dtype at(Index i) const { return m_mat(i); }
    inline Dtype& at(Index i) { return m_mat(i); }
    inline Dtype operator() (Index i) const { return at(i); }
    inline Dtype& operator() (Index i) { return at(i); }

    /**
     * some simple function
     */
    inline Dtype sum() const { return m_mat.sum(); }
    inline Dtype prod() const { return m_mat.prod(); }
    inline Dtype min() const { return m_mat.minCoeff(); }
    inline Dtype max() const { return m_mat.maxCoeff(); }
    inline Dtype trace() const { return m_mat.trace(); }

    /**
     * the print functions
     */
    inline friend std::ostream& operator<<(std::ostream &os, const EMatrix<Dtype>& e) {
        os << e.m_mat;
        return os;
    }

private:
    //the contranier of matrix datas
    RawMatrix<Dtype> m_mat;

private:

    void InitByValue(Dtype val) {
        Index length = m_mat.rows() * m_mat.cols();
        for(Index i = 0; i < length; ++i) {
            m_mat(i) = val;
        }
    }

public:
    // the function to apply element wise operation
    template<typename Fun, typename... Args>
    void element_apply(Index i, Index j, Index p, Index q, Fun&& fun, Args&&... args) {
        assert(i >= 0 && j >= 0 && i + p < rows() && j + q < cols());
        for(Index ii = i; ii < i + p; ++ii) {
            for(Index jj = j; jj < j + q; ++jj) {
                m_mat(ii, jj) = fun(std::forward<Args>(args)...);
            }
        }
    }

};


} //wzp


#endif /*WZP_EIGEN_MATRIX_HPP_*/