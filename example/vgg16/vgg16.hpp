#ifndef VGG16_HPP
#define VGG16_HPP

#include "licon/licon.hpp"

using namespace licon;

// get the conv block
nn::NodePtr ConvBlock(int in_channel, int out_channel, bool use_ave = false) {
    auto block = nn::Squential::CreateSquential();
    block->Add(nn::Conv::CreateConv(in_channel, out_channel, 3, 1, 1));
    block->Add(nn::Relu::CreateRelu());
    block->Add(nn::Conv::CreateConv(out_channel, out_channel, 3, 1, 1));
    block->Add(nn::Relu::CreateRelu());
    if(use_ave) block->Add(nn::AvePool::CreateAvePool(2));
    else block->Add(nn::MaxPool::CreateMaxPool(2));
    return block;
}

nn::Model Vgg() {
    auto block = nn::Squential::CreateSquential();
    block->Add(ConvBlock(3, 32));
    block->Add(ConvBlock(32, 32, true));
    block->Add(ConvBlock(32, 64, true));
    block->Add(nn::Linear::CreateLinear(4 * 4 * 64, 64));
    block->Add(nn::Linear::CreateLinear(64, 10));
    return block;
}

#endif