#ifndef LICON_UTILS_EVALUATION_HPP_
#define LICON_UTILS_EVALUATION_HPP_

#include <memory>
#include <vector>

#include "licon/common.hpp"

/**
 * the interface to eval different type's model
 */
namespace licon
{

namespace nn
{
// the decalaration of opnode
template<typename Dtype>
class OpNode;
} //nn

namespace io
{
// the decalaration of dataset
template<typename... Args>
class Dataset;
} //io

namespace utils
{

// the base interface for Evaluation
template<typename... Args>
class Evaluation {
public:
    // the constructor
    Evaluation(std::unique_ptr<nn::OpNode<F> >& model, io::Dataset<Args...>& eval_dataset)
    : m_model(model), m_eval_dataset(eval_dataset)
    {}

    // the run interface
    virtual F Run() = 0;

protected:
    std::unique_ptr<nn::OpNode<F> >& m_model;
    io::Dataset<Args...>& m_eval_dataset;

};

// the accuracy evaluation
class Accuracy : public Evaluation<std::vector<unsigned char>*, int> {

public:
    static std::unique_ptr<Evaluation<std::vector<unsigned char>*, int> > CreateAccuracy(
            std::unique_ptr<nn::OpNode<F> >& model, io::Dataset<std::vector<unsigned char>*, int>& eval_dataset);

    F Run();

private:
    Accuracy(std::unique_ptr<nn::OpNode<F> >& model, io::Dataset<std::vector<unsigned char>*, int>& eval_dataset);

};


} //utils


} //licon


#endif /*LICON_UTILS_EVALUATION_HPP_*/