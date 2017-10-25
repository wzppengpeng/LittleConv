#include "licon/nn/node/conv.hpp"

#include "licon/nn/init.hpp"


#include "licon/utils/ematrix_args.hpp"

#include "licon/utils/transformer.hpp"

#include "function/help_function.hpp"

using namespace std;


const static F BIAS_VAL = 0.;

namespace licon
{

namespace nn
{

std::unordered_map<std::string, utils::ETensor<F>* > Conv::StateDict() {
    CHECK(m_node_name.empty() == false);
    std::unordered_map<std::string, utils::ETensor<F>* > state_dict_linear;
    state_dict_linear.emplace(m_node_name + "W", weight());
    state_dict_linear.emplace(m_node_name + "B", bias());
    return state_dict_linear;
}

void Conv::LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict) {
    CHECK(m_node_name.empty() == false);
    ASSERT(state_dict.find(m_node_name + "W") != state_dict.end(), "Miss Conv Weight", m_node_name);
    ASSERT(state_dict.find(m_node_name + "B") != state_dict.end(), "Miss Conv Bias", m_node_name);
    m_weights = state_dict.find(m_node_name + "W")->second;
    m_bias = state_dict.find(m_node_name + "B")->second;
}

nn::NodePtr Conv::CreateConv(int in_channels, int out_channels, int kernel_size, int stride,
        int padding, int dilation) {
    return std::unique_ptr<OpNode<F> >(new Conv(in_channels, out_channels, kernel_size, stride,
        padding, dilation));
}

nn::NodePtr Conv::CreateConv(int in_channels, int out_channels, std::pair<int, int> kernel_size,
        std::pair<int, int> stride, std::pair<int, int> padding,
        std::pair<int, int> dilation) {
    return std::unique_ptr<OpNode<F> >(new Conv(in_channels, out_channels, std::move(kernel_size),
        std::move(stride), std::move(padding), std::move(dilation)));
}

// constructors
Conv::Conv(int in_channels, int out_channels, int kernel_size, int stride,
        int padding, int dilation) {
    ConvSetUp(in_channels, out_channels, {kernel_size, kernel_size}, {stride, stride},
        {padding, padding}, {dilation, dilation});
}

Conv::Conv(int in_channels, int out_channels, std::pair<int, int> kernel_size,
        std::pair<int, int> stride, std::pair<int, int> padding,
        std::pair<int, int> dilation) {
    ConvSetUp(in_channels, out_channels, std::move(kernel_size), std::move(stride), std::move(padding),
        std::move(dilation));
}


// set up the conv node
void Conv::ConvSetUp(int in_channels, int out_channels, std::pair<int, int> kernel_size,
        std::pair<int, int> stride, std::pair<int, int> padding,
        std::pair<int, int> dilation) {
    // the check assert
    ASSERT(in_channels > 0 && out_channels > 0, "the input and output channels should br higher than zero");
    ASSERT(kernel_size.first > 0 && kernel_size.second > 0, "the kernel size should be higher than zero");
    ASSERT(stride.first > 0 && stride.second > 0, "the stride should be higher than zero");
    ASSERT(padding.first >= 0 && padding.second >= 0, "the padding param should not be negative");
    ASSERT(dilation.first > 0 && dilation.second > 0, "the dilation should be higher than zero");
    NeuronNode<F>::Resize();
    m_conv_in_channels = in_channels;
    m_kernel_shape[0] = kernel_size.first;
    m_kernel_shape[1] = kernel_size.second;
    m_stride[0] = stride.first;
    m_stride[1] = stride.second;
    m_padding[0] = padding.first;
    m_padding[1] = padding.second;
    m_dilation[0] = dilation.first;
    m_dilation[1] = dilation.second;
    m_is_1x1 = (m_kernel_shape[0] == 1 && m_kernel_shape[1] == 1);
    m_num_output = out_channels;
    m_conv_out_channels = out_channels;
    // set up the weights and bias
    std::vector<int> weight_shape = {m_conv_out_channels, m_conv_in_channels, m_kernel_shape[0], m_kernel_shape[1]};
    vector<int> bias_shape = {1, m_num_output, 1, 1};
    // reshape the weight and bias
    m_weights.Reshape(weight_shape);
    m_bias.Reshape(bias_shape); //need init here
    m_weights_grad.Reshape(weight_shape);
    m_bias_grad.Reshape(bias_shape);
    InitParameters();
    m_kernel_dim = m_weights.count(1);
    m_weight_offset = m_conv_out_channels * m_kernel_dim;
}

void Conv::InitParameters() {
    if(!m_has_initialzed) {
        nn::ParamInit<F>::xavier_uniform(m_weights);
        nn::ParamInit<F>::constant(m_bias, BIAS_VAL);
    }
    m_has_initialzed = true;
}

std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > Conv::RegisterWeights() {
    return {{&m_weights, &m_weights_grad}, {&m_bias, &m_bias_grad}};
}

void Conv::ReshapeForward(const std::vector<utils::ETensor<F>* >& bottom) {
    m_bottom_data = bottom[0];
    m_num = m_bottom_data->num();
    ASSERT(m_bottom_data->channel() == m_conv_in_channels, "the input channels should be same", m_conv_in_channels);
    // get the output shape
    ComputeOuputShape();
    // reshape the data output
    m_data.Reshape(m_num, m_num_output, m_output_shape[0], m_output_shape[1]); //n * Cout * Hout * Wout
    m_conv_out_spatial_dim = m_data.count(2); //2 * 2
    m_col_offset = m_kernel_dim * m_conv_out_spatial_dim; // 3 * 3 * 3 * 4
    m_output_offset = m_conv_out_channels * m_conv_out_spatial_dim; // 4 * 2 * 2
    m_conv_input_shape[0] = m_bottom_data->height();
    m_conv_input_shape[1] = m_bottom_data->width();
    // The im2col result buffer will only hold one image at a time to avoid
    // overly large memory usage. In the special case of 1x1 convolution
    // it goes lazily unused to save memory.
    m_col_buffer_shape.clear();
    m_col_buffer_shape.emplace_back(1);
    m_col_buffer_shape.emplace_back(m_kernel_dim);
    m_col_buffer_shape.emplace_back(m_output_shape[0]);
    m_col_buffer_shape.emplace_back(m_output_shape[1]);
    m_col_buffer.Reshape(m_col_buffer_shape);
    m_bottom_dim = m_bottom_data->count(1);
    m_top_dim = m_data.count(1);
    m_num_kernels_im2col = m_conv_in_channels * m_conv_out_spatial_dim; //3 * 2 * 2
    m_num_kernels_col2im = m_bottom_dim; // c* h * w
    m_out_spatial_dim = m_data.count(2);
    // bias_multiplier_ need not
}

void Conv::ReshapeBackward(const std::vector<utils::ETensor<F>* >& top) {
    ASSERT(top[0]->num() == m_data.num() && top[0]->channel() == m_data.channel()
        && top[0]->height() == m_data.height() && top[0]->width() == m_data.width(),
        "Shape Miss Match");
    m_grad.Reshape(m_bottom_data->shape());
}

void Conv::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    ReshapeForward(bottom);
    // for each mini batch, only handle one image per conv
    for(int i = 0; i < m_num; ++i) {
        // weight gemm
        ForwardGemm(m_bottom_data->ptr(i), m_data.mutable_ptr(i));
        // bias add
        ForwardBias(m_data.mutable_ptr(i), m_bias.mutable_ptr());
    }
}

void Conv::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ReshapeBackward(top);
    auto& top_grad = *top[0];
    for(int i = 0; i < m_num; ++i) {
        // grad backward
        BackwardGemm(top_grad.mutable_ptr(i), m_grad.mutable_ptr(i));
        // weight grad backward
        WeightGemm(m_bottom_data->ptr(i), top_grad.mutable_ptr(i));
        // bias grad backward
        BackwardBias(top_grad.mutable_ptr(i));
    }
    m_bottom_data = nullptr;
}

void Conv::ComputeOuputShape() {
    // compute the new height
    m_output_shape[0] = (m_bottom_data->height() + 2 * m_padding[0] - (m_dilation[0] * (m_kernel_shape[0] - 1) + 1)) / m_stride[0] + 1;
    m_output_shape[1] = (m_bottom_data->width() + 2 * m_padding[1] - (m_dilation[1] * (m_kernel_shape[1] - 1) + 1)) / m_stride[1] + 1;
}

void Conv::ForwardGemm(const F* input, F* output) {
    // get the im2col
    wzp::EMatrix<F> col_matrix(m_kernel_dim, m_conv_out_spatial_dim, m_col_buffer.mutable_ptr());
    conv_im2col(input, col_matrix);
    auto res = (m_weights.ViewInPlace<wzp::EMatrix<F> >()) * col_matrix;
    // copy the matrix data into output pointer
    utils::EMatrixArgs<F>::copy_to_pointer(res, output);
}

void Conv::ForwardBias(F* output, F* bias) {
    // transpose the output pointer into matrix
    wzp::EMatrix<F> out_maitrx(m_num_output, m_out_spatial_dim, output); // 3 * 4
    wzp::EMatrix<F> bias_vector(m_num_output, 1, bias); //3 * 1
    out_maitrx += bias_vector; //add to the output pointer
}

void Conv::BackwardGemm(F* output, F* input) {
    // weight * output
    wzp::EMatrix<F> output_matrix(m_conv_out_channels, m_conv_out_spatial_dim, output);
    auto grad_res = (m_weights.ViewInPlace<wzp::EMatrix<F>>().transpose()) * output_matrix; //27 * 4
    // copy the grad res to col buffer
    utils::EMatrixArgs<F>::copy_to_pointer(grad_res, m_col_buffer.mutable_ptr());
    // col2im
    wzp::EMatrix<F> col_matrix(m_kernel_dim, m_conv_out_spatial_dim, m_col_buffer.mutable_ptr());
    conv_col2im(col_matrix, input);
}

void Conv::WeightGemm(const F* input, F* output) {
    // get the im2col
    wzp::EMatrix<F> col_matrix(m_kernel_dim, m_conv_out_spatial_dim, m_col_buffer.mutable_ptr());
    conv_im2col(input, col_matrix);
    wzp::EMatrix<F> output_matrix(m_conv_out_channels, m_conv_out_spatial_dim, output);
    // here use += to add grad, because only handle one image
    auto weight_grad = output_matrix * (col_matrix.transpose()); //3 * 27
    // add the weight grad to the weight grad total
    utils::EMatrixArgs<F>::add_to_pointer(weight_grad, m_weights_grad.mutable_ptr());
}

void Conv::BackwardBias(F* input) {
    wzp::EMatrix<F> grad_matrix(m_conv_out_channels, m_conv_out_spatial_dim, input);
    auto bias_grad = grad_matrix * (wzp::EMatrix<F>(m_conv_out_spatial_dim, 1, 1.0)); //3 * 1
    utils::EMatrixArgs<F>::add_to_pointer(bias_grad, m_bias_grad.mutable_ptr());
}

} //nn

} //licon