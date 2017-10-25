#ifndef LICON_UTILS_IM_2_COL_HPP_
#define LICON_UTILS_IM_2_COL_HPP_


/**
 * the template function of im2col
 * transpose the C * H * W feature map into the matrix type
 * the template function of col2im
 * transpose the matrix into C * H * W feature map(grad)
 */
namespace licon
{

namespace utils
{

// check the zero and negative location
inline static bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// set the point values
template<typename Dtype>
inline static void set_pointer_value(int len, Dtype* ptr, Dtype val) {
    for(int i = 0; i < len; ++i) {
        ptr[i] = val;
    }
}

// this code is copy from caffe~~
template<typename Dtype, typename Mat>
void im2col(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Mat& mat) {
    int idx = 0;
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for(int channel = channels; channel--; data_im += channel_size) {
        for(int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
            for(int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
                // the location of input row
                int input_row = -pad_h + kernel_row * dilation_h;
                for(int output_rows = output_h; output_rows; --output_rows) {
                    if(!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        for(int output_cols = output_w; output_cols; --output_cols) {
                            mat(idx++) = Dtype(0);
                        }
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for(int output_cols = output_w; output_cols; --output_cols) {
                            if(is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                mat(idx++) = data_im[input_row * width + input_col];
                            } else {
                                mat(idx++) = Dtype(0);
                            }
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}


// this code is also copied from caffe
template<typename Dtype, typename Mat>
void col2im(const Mat& mat, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_im) {
    // set the data im to zero
    set_pointer_value(height * width * channels, data_im, Dtype(0));
    int idx = 0;
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    for(int channel = channels; channel--; data_im += channel_size) {
        for(int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
            for(int kernel_col = 0; kernel_col < kernel_w; ++kernel_col) {
                int input_row = -pad_h + kernel_row * dilation_h;
                for(int output_rows = output_h; output_rows; --output_rows) {
                    if(!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        idx += output_w;
                    } else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for(int output_cols = output_w; output_cols; --output_cols) {
                            if(is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                data_im[input_row * width + input_col] += mat(idx); //for the grad need to add together
                                // data_im[input_row * width + input_col] = mat(idx); //????????
                            }
                            ++idx;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }
            }
        }
    }
}


} //utils

} //licon



#endif /*LICON_UTILS_IM_2_COL_HPP_*/