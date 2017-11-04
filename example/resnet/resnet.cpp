/**
 * the example of resnet
 */

#include "config/click.hpp"

#include "model.hpp"

nn::Model get_model() {
    return ResNet18();
}

void train() {
    std::string input_dir = wzp::Click::get("--input_dir");
    // the parameters
    int batch_size = 256;
    int epoch_num = 50;
    float lr = 1e-2;
    int display = 50;
    float weight_deacay = 5e-5;

    // get model
    auto model = get_model();
    model->set_node_name("cifar10_resnet18");

     // define the loss
    auto cross_entropy_loss = nn::CrossEntropyLoss::CreateCrossEntropyLoss();

    // define the optimizer
    auto optimizer = optim::Adam::CreateAdam(model->RegisterWeights(), lr, 0.9, 0.999, weight_deacay);
    auto scheduler = optim::StepLR::CreateStepLR(*optimizer.get(), 5, 0.95);

    io::Cifar10Dataset cifar10_train(input_dir, io::Cifar10Dataset::TRAIN);
    io::Cifar10Dataset cifar10_test(input_dir, io::Cifar10Dataset::TEST);
    cifar10_train.Load();
    cifar10_test.Load();

    // define a trainer to train the classification task
    auto trainer = utils::ClassifyTrainer::CreateClassfyTrainer(model, cross_entropy_loss, optimizer, cifar10_train,
        batch_size, epoch_num, display, &cifar10_test, &scheduler);

    // train
    trainer->Train();
    // save model
    io::Saver::Save("cifar10_resnet18.liconmodel", model);
}


void test() {
    std::string input_dir = wzp::Click::get("--input_dir");
    // get model
    auto model = get_model();
    model->test();

    // load model
    io::Saver::Load("cifar10_resnet18.liconmodel", &model);
    wzp::log::info("cifar10_resnet18 model loaded...");

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

    wzp::log::info("Run Cifar10 Resnet18...");

    if(wzp::Click::get("--mode") == "train") {
        train();
    } else {
        test();
    }
    return 0;
}

