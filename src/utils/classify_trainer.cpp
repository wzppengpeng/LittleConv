#include "licon/utils/trainer.hpp"

#include "licon/nn/operation_node.hpp"
#include "licon/io/dataset.hpp"
#include "licon/optim/optim.hpp"

#include "licon/utils/evaluation.hpp"

#include "licon/utils/etensor_args.hpp"

#include "licon/utils/transformer.hpp"

#include "log/log.hpp"
#include "util/timer.hpp"

#include "function/help_function.hpp"

namespace licon
{


namespace utils
{


// the implemention of ClassifyTrainer
class ClassifyTrainerImpl : public ClassifyTrainer {
public:
    ClassifyTrainerImpl(std::unique_ptr<nn::OpNode<F> >& model,
                        std::unique_ptr<nn::OpNode<F> >& loss_node,
                        std::unique_ptr<optim::Optimizer>& optimizer,
                        io::Dataset<std::vector<unsigned char>*, int>& train_dataset,
                        int batch_size,
                        int epoch,
                        int display,
                        io::Dataset<std::vector<unsigned char>*, int>* validation_dataset = nullptr);

    void Train();

private:
    std::unique_ptr<nn::OpNode<F> >& m_model;
    std::unique_ptr<nn::OpNode<F> >& m_loss_node;
    std::unique_ptr<optim::Optimizer>& m_optimizer;
    io::Dataset<std::vector<unsigned char>*, int>& m_train_dataset;
    int m_batch_size;
    int m_epoch;
    int m_display;
    io::Dataset<std::vector<unsigned char>*, int>* m_validation_dataset;

private:
    void TrainIter(int iter);

    F Eval(io::Dataset<std::vector<unsigned char>*, int>& dataset);

};

ClassifyTrainerImpl::ClassifyTrainerImpl(std::unique_ptr<nn::OpNode<F> >& model,
                        std::unique_ptr<nn::OpNode<F> >& loss_node,
                        std::unique_ptr<optim::Optimizer>& optimizer,
                        io::Dataset<std::vector<unsigned char>*, int>& train_dataset,
                        int batch_size,
                        int epoch,
                        int display,
                        io::Dataset<std::vector<unsigned char>*, int>* validation_dataset)

                    : m_model(model), m_loss_node(loss_node), m_optimizer(optimizer), m_train_dataset(train_dataset),
                    m_batch_size(batch_size), m_epoch(epoch), m_display(display), m_validation_dataset(validation_dataset)
{}


void ClassifyTrainerImpl::TrainIter(int iter) {
    m_model->set_phase(Phase::TRAIN);
    io::MnistCifar10Loader loader(&m_train_dataset, true, m_batch_size);
    wzp::Timer t;
    F mean_loss = 0.0, mean_accuracy = 0.0;
    int total = 0;
    int cnt = 0;
    while(loader.has_next()) {
        auto data_label = loader.next();
        utils::ETensor<F>* input_data = &data_label.first;
        utils::ETensor<F> input_label;
        utils::transform_vector_to_tensor(input_label, data_label.second);
        // forward
        m_model->Forward({input_data});
        m_loss_node->Forward({m_model->data(), &input_label});
        mean_loss += m_loss_node->data()->at(0);
        // backward
        auto out = utils::ETensorArgs<F>::generate_scalar(1);
        m_loss_node->Backward({&out});
        m_model->Backward({m_loss_node->grad()});
        // update the parameter
        m_optimizer->Step();
        ++total;
        if(++cnt % m_display == 0) {
            wzp::log::info("TRAIN INFOMATION", "Iter", iter, "Epoch", m_epoch, "MiniIter", cnt, "loss", m_loss_node->data()->at(0));
        }
    }
    mean_loss /= total;
    auto used_time = t.elapsed_seconds();
    mean_accuracy = Eval(m_train_dataset);
    if(m_validation_dataset != nullptr) {
        F validation_accuracy = Eval(*m_validation_dataset);
        wzp::log::info("TRAIN INFOMATION", "Iter", iter, "Epoch", m_epoch, "used time(s)", used_time,
            "loss", mean_loss, "train_accuracy", mean_accuracy, "validation_accuracy", validation_accuracy);
    } else {
        wzp::log::info("TRAIN INFOMATION", "Iter", iter, "Epoch", m_epoch, "used time(s)", used_time,
            "loss", mean_loss, "train_accuracy", mean_accuracy);
    }
}

F ClassifyTrainerImpl::Eval(io::Dataset<std::vector<unsigned char>*, int>& dataset) {
    auto eval = utils::Accuracy::CreateAccuracy(m_model, dataset);
    return eval->Run();
}

void ClassifyTrainerImpl::Train() {
    wzp::log::info("Begin to train epoch number", m_epoch, "batch size", m_batch_size);
    for(int i = 1; i <= m_epoch; ++i) {
        TrainIter(i);
    }
    wzp::log::info("Train Pocess Finished...");
}



std::unique_ptr<Trainer> ClassifyTrainer::CreateClassfyTrainer(std::unique_ptr<nn::OpNode<F> >& model,
                                                         std::unique_ptr<nn::OpNode<F> >& loss_node,
                                                         std::unique_ptr<optim::Optimizer>& optimizer,
                                                         io::Dataset<std::vector<unsigned char>*, int>& train_dataset,
                                                         int batch_size,
                                                         int epoch,
                                                         int display,
                                                         io::Dataset<std::vector<unsigned char>*, int>* validation_dataset)
{
    return std::unique_ptr<Trainer>(new ClassifyTrainerImpl(model, loss_node, optimizer, train_dataset, batch_size, epoch, display, validation_dataset));
}

} //utils

} //licon

using namespace std;