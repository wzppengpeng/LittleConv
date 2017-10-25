#ifndef LICON_NN_NODE_CONV_HPP_
#define LICON_NN_NODE_CONV_HPP_


/**
 * the conv layer. to make if eaiser, set the group num to be 1
 */

#include <array>

#include <mutex>

#include "licon/nn/node/neuron.hpp"

#include "licon/utils/im2col.hpp"


namespace wzp
{

template<typename T>
class EMatrix;

} //wzp

namespace licon
{

namespace nn
{

// the convolution node
class Conv : public NeuronNode<F> {
public:
    // the heap create function
    static nn::NodePtr CreateConv(int in_channels, int out_channels, int kernel_size, int stride = 1,
        int padding = 0, int dilation = 1);

    static nn::NodePtr CreateConv(int in_channels, int out_channels, std::pair<int, int> kernel_size,
        std::pair<int, int> stride, std::pair<int, int> padding,
        std::pair<int, int> dilation);

    // the node's name(type)
    virtual inline std::string name() const { return "conv"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

    // init params
    virtual void InitParameters();

    // register parameters
    virtual std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > RegisterWeights();


    // get the weight and bias point
    virtual inline utils::ETensor<F>* weight() { return &m_weights; }
    virtual inline utils::ETensor<F>* bias() { return &m_bias; }

    // the state dict getter and loader
    virtual std::unordered_map<std::string, utils::ETensor<F>* > StateDict();
    virtual void LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict);

protected:
    // the pre saved input data
    utils::ETensor<F>* m_bottom_data = nullptr;

    // the conv paramethers(height_param, width_param)
    std::array<int, 2> m_kernel_shape;
    std::array<int, 2> m_stride;
    std::array<int, 2> m_padding;
    std::array<int, 2> m_dilation;
    // the input spatial shape of input convolution
    std::array<int, 2> m_conv_input_shape;
    // the col buffer shape of spatial
    std::vector<int> m_col_buffer_shape;
    // the spatial dimension of output
    std::array<int, 2> m_output_shape;

    int m_num;
    // the group num
    // int m_group;
    int m_out_spatial_dim;
    int m_weight_offset;
    int m_num_output;
    bool m_is_1x1;

    // the col buffer and bias multiplier
    //use multithread, so each thread will allocate new memory

    // the parameters
    utils::ETensor<F> m_weights; //input * output
    utils::ETensor<F> m_bias; //1 * output

    // save the weights grad
    utils::ETensor<F> m_weights_grad;
    utils::ETensor<F> m_bias_grad;

    // have been init
    bool m_has_initialzed = false;


private:
    // the constructor
    Conv(int in_channels, int out_channels, int kernel_size, int stride = 1,
        int padding = 0, int dilation = 1);
    // create with different shape in different height and width
    Conv(int in_channels, int out_channels, std::pair<int, int> kernel_size,
        std::pair<int, int> stride, std::pair<int, int> padding,
        std::pair<int, int> dilation);

    void ConvSetUp(int in_channels, int out_channels, std::pair<int, int> kernel_size,
        std::pair<int, int> stride, std::pair<int, int> padding,
        std::pair<int, int> dilation);

    // some temp varibales
    int m_num_kernels_im2col;
    int m_num_kernels_col2im;
    int m_conv_out_channels;
    int m_conv_in_channels;
    int m_conv_out_spatial_dim;
    int m_kernel_dim;
    int m_col_offset;
    int m_output_offset;

    int m_bottom_dim;
    int m_top_dim;

    mutable std::mutex m_mut;

protected:
    //functions
    // forward reshape the data
    void ReshapeForward(const std::vector<utils::ETensor<F>* >& bottom);
    // backward reshape the grad
    void ReshapeBackward(const std::vector<utils::ETensor<F>* >& top);

    // wrap im2col
    template<typename Mat>
    inline void conv_im2col(const F* data, Mat& mat) {
        utils::im2col(data, m_conv_in_channels, m_conv_input_shape[0], m_conv_input_shape[1],
            m_kernel_shape[0], m_kernel_shape[1],
            m_padding[0], m_padding[1],
            m_stride[0], m_stride[1],
            m_dilation[0], m_dilation[1], mat);
    }

    // wrap col2im
    template<typename Mat>
    inline void conv_col2im(const Mat& mat, F* data) {
        utils::col2im(mat, m_conv_in_channels, m_conv_input_shape[0], m_conv_input_shape[1],
            m_kernel_shape[0], m_kernel_shape[1],
            m_padding[0], m_padding[1],
            m_stride[0], m_stride[1],
            m_dilation[0], m_dilation[1], data);
    }

    // compute the output shape
    void ComputeOuputShape();

    // helper functions for gemm
    void ForwardGemm(const F* input, F* output);
    void ForwardBias(F* output, F* bias);
    void BackwardGemm(F* output, F* input);
    void WeightGemm(const F* input, F* output);
    void BackwardBias(F* input);

    void WriteWeight(wzp::EMatrix<F>* mat);
    void WriteBias(wzp::EMatrix<F>* mat);

};

} //nn

} //licon

#endif /*LICON_NN_NODE_CONV_HPP_*/