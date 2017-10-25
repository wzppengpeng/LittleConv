#ifndef LICON_UTILS_TRAINER_HPP_
#define LICON_UTILS_TRAINER_HPP_

#include <memory>
#include <vector>

#include "licon/common.hpp"

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

// the declaration of dataset
template<typename... Args>
class Dataset;

} //io

namespace optim
{
// the decalaration of optimizer
class Optimizer;
} //optim

namespace utils
{

// the base class of Trainer
class Trainer {

public:
    virtual void Train() = 0;

};



// the base class of classify trainer
class ClassifyTrainer : public Trainer {
public:
    // the trainer for classification
    static std::unique_ptr<Trainer> CreateClassfyTrainer(std::unique_ptr<nn::OpNode<F> >& model,
                                                         std::unique_ptr<nn::OpNode<F> >& loss_node,
                                                         std::unique_ptr<optim::Optimizer>& optimizer,
                                                         io::Dataset<std::vector<unsigned char>*, int>& train_dataset,
                                                         int batch_size,
                                                         int epoch,
                                                         int display = 1000,
                                                         io::Dataset<std::vector<unsigned char>*, int>* validation_dataset = nullptr);

    virtual void Train() = 0;

};

} //utils

} //licon


#endif /*LICON_UTILS_TRAINER_HPP_*/