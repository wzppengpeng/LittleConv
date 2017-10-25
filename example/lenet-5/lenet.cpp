/**
 * the example to run lenet-5
 */
#include "licon/licon.hpp"

#include "config/click.hpp"

using namespace std;
using namespace licon;

nn::Model get_model() {
    auto block = nn::Squential::CreateSquential();
    block->Add(nn::Conv::CreateConv(1, 20, 5));
    block->Add(nn::Relu::CreateRelu());
    block->Add(nn::MaxPool::CreateMaxPool(2)); //12 * 12

    block->Add(nn::Conv::CreateConv(20, 50, 5));
    block->Add(nn::Relu::CreateRelu());
    block->Add(nn::MaxPool::CreateMaxPool(2)); //4 * 4

    block->Add(nn::Linear::CreateLinear(4 * 4 * 50, 500));
    block->Add(nn::Relu::CreateRelu());
    block->Add(nn::Linear::CreateLinear(500, 10));
    block->Add(nn::Softmax::CreateSoftmax());
    return block;
}

void train() {
    // the parameters
    int batch_size = 256;
    int epoch_num = 10;
    float lr = 1e-4;
    int display = 150;

    std::string input_dir = wzp::Click::get("--input_dir");
    // get model
    auto model = get_model();

    model->set_node_name("Lenet-5");

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
    // save model
    io::Saver::Save("lenet-5.liconmodel", model);
}

void test() {
    std::string input_dir = wzp::Click::get("--input_dir");;
    auto model = get_model();
    model->set_phase(licon::Phase::TEST);

    // load model
    io::Saver::Load("lenet-5.liconmodel", &model);
    wzp::log::info("lenet-5 model loaded...");

    io::MnistDataset mnist_test(input_dir, io::MnistDataset::TEST);
    mnist_test.Load();
    // eval
    wzp::log::info("The final accuracy is", utils::Accuracy::CreateAccuracy(model, mnist_test)->Run());
}

int main(int argc, char *argv[])
{
    wzp::Click::add_argument("--mode", "the mode train or test");
    wzp::Click::add_argument("--input_dir", "the input dir of mnist root dir");
    wzp::Click::parse(argc, argv);

    // set up the env
    licon::EnvSetUp();

    wzp::log::info("Run Mnist Lenet-5...");

    if(wzp::Click::get("--mode") == "train") {
        train();
    } else {
        test();
    }


    return 0;
}