#include "licon/nn/node/max_pool.hpp"

#include "licon/utils/etensor_args.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

// the temp connection char sysbom
#ifndef O
#define O 'o'
#endif //O

#ifndef X
#define X 'x'
#endif //X

namespace licon
{

namespace nn
{

/**
 * details functions
 */
namespace max_pool
{

// chech the input shape
inline static void check_input_feature_map(utils::ETensor<F>* bottom_map, int kernel_size) {
    ASSERT(bottom_map->height() % kernel_size == 0, "The kernal size is mismatch height", kernel_size);
    ASSERT(bottom_map->width() % kernel_size == 0, "The kernal size is mismatch width", kernel_size);
}

// get the max value of input data and record into mask
F compute_window_max(const utils::ETensor<F>& bottom_data, int n, int c, int h_high, int w_high, int kernel_size, utils::ETensor<char>& mask) {
    int h_low = h_high * kernel_size;
    int w_low = w_high * kernel_size;
    F max_val = bottom_data(n, c, h_low, w_low);
    int h_max_loc = h_low, w_max_loc = w_low;
    for(int h = h_low; h < h_low + kernel_size; ++h) {
        for(int w = w_low; w < w_low + kernel_size; ++w) {
            if(bottom_data(n, c, h, w) > max_val) {
                h_max_loc = h; w_max_loc = w;
                max_val = bottom_data(n, c, h, w);
            }
        }
    }
    mask(n, c, h_max_loc, w_max_loc) = X;
    return max_val;
}

// set the window's grad depended on the mask
void compute_window_grad(const utils::ETensor<F>& top_grad, int n, int c, int h_high, int w_high, int kernel_size, const utils::ETensor<char>& mask, utils::ETensor<F>& grad) {
    int h_low = h_high * kernel_size;
    int w_low = w_high * kernel_size;
    for(int h = h_low; h < h_low + kernel_size; ++h) {
        for(int w = w_low; w < w_low + kernel_size; ++w) {
            if(mask(n, c, h, w) == X) {
                grad(n, c, h, w) = top_grad(n, c, h_high, w_high);
            } else {
                grad(n, c, h, w) = 0.0;
            }
        }
    }
}

} //max_pool

std::unique_ptr<OpNode<F> > MaxPool::CreateMaxPool(const int kernel_size) {
    return std::unique_ptr<OpNode<F> >(new MaxPool(kernel_size));
}

void MaxPool::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    max_pool::check_input_feature_map(bottom[0], m_kernel_size);
    // reshape the output map
    m_data.Reshape(bottom[0]->num(), bottom[0]->channel(), bottom[0]->height() / m_kernel_size, bottom[0]->width() / m_kernel_size);
    // reshape the mask
    m_mask.Reshape(bottom[0]->shape());
    utils::ETensorArgs<char>::fill(m_mask, O);
    auto& bottom_data = *(bottom[0]);
    // forward and record the max area
    wzp::ParallelRange(bottom_data.num(), [&bottom_data, this](int n) {
        for(int c = 0; c < bottom_data.channel(); ++c) {
            for(int h_high = 0; h_high < m_data.height(); ++h_high) {
                for(int w_high = 0; w_high < m_data.width(); ++w_high) {
                    m_data(n, c, h_high, w_high) =  max_pool::compute_window_max(bottom_data, n, c, h_high, w_high, m_kernel_size, m_mask);
                }
            }
        }
    });
    m_has_forwarded = true;
}

void MaxPool::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_has_forwarded);
    check_backward(top);
    auto& top_grad = *(top[0]);
    // reshape the grad
    m_grad.Reshape(top_grad.num(), top_grad.channel(), top_grad.height() * m_kernel_size, top_grad.width() * m_kernel_size);
    wzp::ParallelRange(m_grad.num(), [this, &top_grad](int n) {
        for(int c = 0; c < m_grad.channel(); ++c) {
            for(int h_high = 0; h_high < top_grad.height(); ++h_high) {
                for(int w_high = 0; w_high < top_grad.width(); ++w_high) {
                    max_pool::compute_window_grad(top_grad, n, c, h_high, w_high, m_kernel_size, m_mask, m_grad);
                }
            }
        }
    });
    m_has_forwarded = false;
}

MaxPool::MaxPool(const int kernel_size) : PoolNode<F>(kernel_size), m_has_forwarded(false) {
    NeuronNode<F>::Resize();
}

} //nn

} //licon

