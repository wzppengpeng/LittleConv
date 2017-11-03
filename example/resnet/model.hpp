/**
 * the model define by resnet
 */

#include "licon/licon.hpp"

using namespace std;
using namespace licon;


nn::NodePtr BasicBlock(int in_channel, int out_channel, int stride=1) {
    auto basic_block = nn::Squential::CreateSquential();
    auto conv_block = nn::Squential::CreateSquential();
    conv_block->Add(nn::Conv::CreateConv(in_channel, out_channel, 3, stride, 1));
    conv_block->Add(nn::BatchNorm::CreateBatchNorm(out_channel));
    conv_block->Add(nn::Relu::CreateRelu());
    conv_block->Add(nn::Conv::CreateConv(out_channel, out_channel, 3, 1, 1));
    conv_block->Add(nn::BatchNorm::CreateBatchNorm(out_channel));
    if(stride == 1 && in_channel == out_channel) {
        auto identity = nn::EltWiseSum::CreateEltWiseSum(true);
        identity->Add(std::move(conv_block));
        basic_block->Add(std::move(identity));
        basic_block->Add(nn::Relu::CreateRelu());
    } else {
        auto identity = nn::EltWiseSum::CreateEltWiseSum(false);
        auto short_cut = nn::Squential::CreateSquential();
        short_cut->Add(nn::Conv::CreateConv(in_channel, out_channel, 1, stride));
        short_cut->Add(nn::BatchNorm::CreateBatchNorm(out_channel));
        identity->Add(std::move(conv_block));
        identity->Add(std::move(short_cut));
        basic_block->Add(std::move(identity));
        basic_block->Add(nn::Relu::CreateRelu());
    }
    return basic_block;
}


nn::Model ResNet(const std::vector<int>& num_blocks, int num_classes=10) {
    auto model = nn::Squential::CreateSquential();
    int in_channel = 32;
    auto _make_layer = [&in_channel](int out_channel, int num_blocks, int stride) {
        vector<int> strides = {stride};
        for(int i = 0; i < num_blocks - 1; ++i) strides.emplace_back(1);
        auto layers = nn::Squential::CreateSquential();
        for(auto s : strides) {
            layers->Add(BasicBlock(in_channel, out_channel, s));
            in_channel = out_channel * 1;
        }
        return layers;
    };
    model->Add(nn::Conv::CreateConv(3, 32, 3, 1, 1));
    model->Add(nn::BatchNorm::CreateBatchNorm(32));
    model->Add(nn::Relu::CreateRelu());
    model->Add(_make_layer(32, num_blocks[0], 1));
    model->Add(_make_layer(64, num_blocks[1], 2)); //16
    model->Add(_make_layer(128, num_blocks[2], 2)); //8
    model->Add(_make_layer(256, num_blocks[3], 2)); //4
    model->Add(nn::AvePool::CreateAvePool(4)); //512 * 1 * 1
    model->Add(nn::Linear::CreateLinear(256, num_classes));
    model->Add(nn::Softmax::CreateSoftmax());
    return model;
}


nn::Model ResNet18() {
    return ResNet({2, 2, 2, 2});
}