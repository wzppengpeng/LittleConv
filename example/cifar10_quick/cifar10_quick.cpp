/**
 * the example of cifar10 quick
 */

#include "licon/licon.hpp"

#include "config/click.hpp"

using namespace std;
using namespace licon;

nn::NodePtr ConvBlock(int in_channel, int out_channel, bool use_ave = false) {
    auto block = nn::Squential::CreateSquential();
    block->Add(nn::Conv::CreateConv(in_channel, out_channel, 5, 1, 2));
    if(use_ave) block->Add(nn::AvePool::CreateAvePool(2));
    else block->Add(nn::MaxPool::CreateMaxPool(2));
    block->Add(nn::Relu::CreateRelu());
    return block;
}

nn::Model get_model() {
    auto block = nn::Squential::CreateSquential();
    block->Add(ConvBlock(3, 32));
    block->Add(ConvBlock(32, 32, true));
    block->Add(ConvBlock(32, 64, true));
    block->Add(nn::Linear::CreateLinear(4 * 4 * 64, 64));
    block->Add(nn::Linear::CreateLinear(64, 10));
    block->Add(nn::Softmax::CreateSoftmax());
    return block;
}

void train() {
    std::string input_dir = wzp::Click::get("--input_dir");
    // the parameters
    int batch_size = 256;
    int epoch_num = 20;
    float lr = 1e-4;
    int display = 100;

    // get model
    auto model = get_model();
    model->set_node_name("cifar10_vgg16");

     // define the loss
    auto cross_entropy_loss = nn::CrossEntropyLoss::CreateCrossEntropyLoss();

    // define the optimizer
    auto optimizer = optim::Adam::CreateAdam(model->RegisterWeights(), lr);

    io::Cifar10Dataset cifar10_train(input_dir, io::Cifar10Dataset::TRAIN);
    io::Cifar10Dataset cifar10_test(input_dir, io::Cifar10Dataset::TEST);
    cifar10_train.Load();
    cifar10_test.Load();

    // define a trainer to train the classification task
    auto trainer = utils::ClassifyTrainer::CreateClassfyTrainer(model, cross_entropy_loss, optimizer, cifar10_train,
        batch_size, epoch_num, display, &cifar10_test);

    // train
    trainer->Train();
    // save model
    io::Saver::Save("cifar10_quick.liconmodel", model);
}


void test() {
    std::string input_dir = wzp::Click::get("--input_dir");
    // get model
    auto model = get_model();
    model->set_phase(licon::Phase::TEST);

    // load model
    io::Saver::Load("cifar10_quick.liconmodel", &model);
    wzp::log::info("cifar10_quick model loaded...");

    io::Cifar10Dataset cifar10_test(input_dir, io::Cifar10Dataset::TEST);
    cifar10_test.Load();
    // eval
    wzp::log::info("The final accuracy is", utils::Accuracy::CreateAccuracy(model, cifar10_test)->Run());
}


int main(int argc, char *argv[])
{
    wzp::Click::add_argument("--mode", "the mode train or test");
    wzp::Click::add_argument("--input_dir", "the input dir of mnist root dir");
    wzp::Click::parse(argc, argv);

    // set up the env
    licon::EnvSetUp();

    wzp::log::info("Run Cifar10 Quick...");

    if(wzp::Click::get("--mode") == "train") {
        train();
    } else {
        test();
    }
    return 0;
}