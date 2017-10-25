/**
 * the example to run the mnist mlp model
 */

#include "licon/licon.hpp"

#include "config/click.hpp"

using namespace std;
using namespace licon;

// the model generator
nn::Model get_model() {
    auto block = nn::Squential::CreateSquential();
    block->Add(nn::Linear::CreateLinear(784, 512));
    block->Add(nn::Relu::CreateRelu(0.2));
    block->Add(nn::Dropout::CreateDropout(0.2));

    block->Add(nn::Linear::CreateLinear(512, 512));
    block->Add(nn::Relu::CreateRelu(0.2));
    block->Add(nn::Dropout::CreateDropout(0.2));

    block->Add(nn::Linear::CreateLinear(512, 10));
    block->Add(nn::Softmax::CreateSoftmax());
    return block;
}


int main(int argc, char  *argv[])
{
    wzp::Click::add_argument("--input_dir", "the input dir of mnist root dir");
    wzp::Click::parse(argc, argv);
    std::string input_dir = wzp::Click::get("--input_dir");

    // set up the env
    licon::EnvSetUp();

    // the parameters
    int batch_size = 256;
    int epoch_num = 20;
    float lr = 1e-3;
    int display = 150;

    // get model
    auto model = get_model();

    // define the loss
    auto cross_entropy_loss = nn::CrossEntropyLoss::CreateCrossEntropyLoss();

    // define the optimizer
    auto optimizer = optim::Adam::CreateAdam(model->RegisterWeights(), lr);

    // load data
    io::MnistDataset mnist(input_dir, io::MnistDataset::TRAIN);
    io::MnistDataset mnist_test(input_dir, io::MnistDataset::TEST);
    mnist.Load();
    mnist_test.Load();

    // define a trainer to train the classification task
    auto trainer = utils::ClassifyTrainer::CreateClassfyTrainer(model, cross_entropy_loss, optimizer, mnist,
        batch_size, epoch_num, display, &mnist_test);

    // train
    trainer->Train();

    model->set_phase(licon::Phase::TEST);

    // eval
    wzp::log::info("The final accuracy is", utils::Accuracy::CreateAccuracy(model, mnist_test)->Run());

    return 0;
}