#ifndef VGG16_HPP
#define VGG16_HPP

#include "licon/licon.hpp"

using namespace licon;

// get the conv block
nn::NodePtr ConvBlock(int in_channel, int out_channel) {
    auto block = nn::Squential::CreateSquential();
    block->Add(nn::Conv::CreateConv(in_channel, out_channel, 3, 1, 1));
    block->Add(nn::Relu::CreateRelu());
    return block;
}

nn::Model Vgg11() {
    auto model = nn::Squential::CreateSquential();
    model->Add(ConvBlock(3, 64));
    model->Add(nn::MaxPool::CreateMaxPool(2)); //16 * 16
    model->Add(ConvBlock(64, 128));
    model->Add(nn::MaxPool::CreateMaxPool(2)); //8 * 8
    model->Add(ConvBlock(128, 256));
    // model->Add(ConvBlock(256, 256));
    model->Add(nn::MaxPool::CreateMaxPool(2)); //4 * 4
    model->Add(ConvBlock(256, 512));
    // model->Add(ConvBlock(512, 512));
    model->Add(nn::MaxPool::CreateMaxPool(2)); //2 * 2
    model->Add(ConvBlock(512, 512));
    model->Add(ConvBlock(512, 512));
    model->Add(nn::MaxPool::CreateMaxPool(2)); //512 * 1 * 1
    return model;
}

#endif