#ifndef VGG16_HPP
#define VGG16_HPP

#include "licon/licon.hpp"

using namespace licon;

// get the conv block
nn::NodePtr ConvBlock(int in_channel, int out_channel, bool triple = false) {
    auto block = nn::Squential::CreateSquential();
    block->Add(nn::Conv::CreateConv(in_channel, out_channel, 3, 1, 1));
    block->Add(nn::BatchNorm::CreateBatchNorm(out_channel));
    block->Add(nn::Relu::CreateRelu());
    block->Add(nn::Conv::CreateConv(out_channel, out_channel, 3, 1, 1));
    block->Add(nn::BatchNorm::CreateBatchNorm(out_channel));
    block->Add(nn::Relu::CreateRelu());
    if(triple) {
        block->Add(nn::Conv::CreateConv(out_channel, out_channel, 3, 1, 1));
        block->Add(nn::BatchNorm::CreateBatchNorm(out_channel));
        block->Add(nn::Relu::CreateRelu());
    }
    block->Add(nn::MaxPool::CreateMaxPool(2));
    return block;
}

nn::Model Vgg() {
    auto net = nn::Squential::CreateSquential();
    net->Add(ConvBlock(3, 8)); //16 * 16
    net->Add(ConvBlock(8, 16)); //8 * 8
    net->Add(ConvBlock(16, 32)); // 4 * 4
    net->Add(ConvBlock(32, 64)); //2 * 2
    net->Add(ConvBlock(64, 128)); //1 * 1
    net->Add(nn::Linear::CreateLinear(128, 10));
    return net;
}

#endif