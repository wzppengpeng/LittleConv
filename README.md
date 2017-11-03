# A simple C++11 deep learning framework, easy to use #
## Introduce ##
LittleConv is a simple deep learning framework based on pure c++11, and does not need anything else.
It is suitable for deep learning on limited computational resource on macos and linux.And it is cpu only.

## Advantage ##
+ *Modern C++*
+ *MultiThread*
+ *Simple Interface*
+ *Easy To Install*

## Dependecy ##
+ g++4.8 or later(with c++11)
+ cmake tools
+ linux or macos

## Compile and Build
It uses cmake to build the project, run shell command as follows:
```
mkdir -p build
cd build
rm -rf *
cmake ..
make -j8
```
The output is a shared library(*.so file*) in the lib directory.

## How To Use
There are some special examples in the example directory such as lenet-5, cifar10-quick.
LittleConv has given interfaces to load mnist and cifar10-bin libraries.
#### How To Build And Run Examples
go to the example directory and run the script(Users need to modify the link_directoies in the CMakeLists.txt)
```
cd example
sh build_example.sh
cd bin
./lenet --input_dir=xxx --mode=train
```
#### How To Define Model
The most important for deep learning framework is defining models.  
LittleConv has an easy way to define models, just like below:
```cpp
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
```
#### How To Train Models
It is also easy to train the model
```cpp
    // the parameters
    int batch_size = 256;
    int epoch_num = 10;
    float lr = 1e-4;
    int display = 150;
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
```
Users only need to define the loss, optimizer and the dataset(mnist or cifar10, other dataset need users to write the dataset reading interface themselves).  
Then pass them to the trainer and call the `Train` interface.  
Actually, the process is very simple. Model has the interface `Forward` and `Backward`, users can DIY the train and test process themselves.  
And the all codes are in the `licon` namespace.  
Please see the example of lenet-5 or cifar10_quick.

## Supported networks
### layer-types and activation functions
+ fully-connected
+ dropout
+ convolution
+ average pooling
+ max pooling
+ softmax
+ tanh
+ sigmoid
+ relu(leaky relu)
+ batch-normalization

### loss functions
+ cross-entropy

### optimization algorithms
+ SGD(with momentum and nesterov)
+ Adam
+ RMSprop

### optimization adjust method
+ LambdaLR
+ StepLR

### layer/operation node container
+ Squential
+ Stack
+ EltWiseSum


## Todo List
+ normalization layers
+ split and concat layers
+ elementwise operation nodes
+ more optimization algorithms





