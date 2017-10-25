#include "licon/nn/node/ave_pool.hpp"

#include "thread/parallel_algorithm.hpp"

using namespace std;

namespace licon
{

namespace nn
{

// the details
namespace ave_pool
{

// chech the input shape
inline static void check_input_feature_map(utils::ETensor<F>* bottom_map, int kernel_size) {
    ASSERT(bottom_map->height() % kernel_size == 0, "The kernal size is mismatch height", kernel_size);
    ASSERT(bottom_map->width() % kernel_size == 0, "The kernal size is mismatch width", kernel_size);
}

// compute the average value of window
F compute_window_ave(const utils::ETensor<F>& bottom_data, int n, int c, int h_high, int w_high, int kernel_size) {
    int h_low = h_high * kernel_size;
    int w_low = w_high * kernel_size;
    F sum_val = 0.0;
     for(int h = h_low; h < h_low + kernel_size; ++h) {
        for(int w = w_low; w < w_low + kernel_size; ++w) {
            sum_val += bottom_data(n, c, h, w);
        }
    }
    return sum_val / static_cast<F>(kernel_size * kernel_size);
}

// compute the average grad backward of window
void compute_window_grad(const utils::ETensor<F>& top_grad, int n, int c, int h_high, int w_high, int kernel_size, utils::ETensor<F>& grad) {
    int h_low = h_high * kernel_size;
    int w_low = w_high * kernel_size;
    F ratio = 1.0 / static_cast<F>(kernel_size * kernel_size);
    for(int h = h_low; h < h_low + kernel_size; ++h) {
        for(int w = w_low; w < w_low + kernel_size; ++w) {
            grad(n, c, h, w) = ratio * top_grad(n, c, h_high, w_high);
        }
    }
}


} //ave pool

std::unique_ptr<OpNode<F> > AvePool::CreateAvePool(const int kernel_size) {
    return std::unique_ptr<OpNode<F> >(new AvePool(kernel_size));
}

void AvePool::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    check_forward(bottom);
    ave_pool::check_input_feature_map(bottom[0], m_kernel_size);
    // reshape the output map
    m_data.Reshape(bottom[0]->num(), bottom[0]->channel(), bottom[0]->height() / m_kernel_size, bottom[0]->width() / m_kernel_size);
    auto& bottom_data = *(bottom[0]);
    // for(int n = 0; n < m_data.num(); ++n) {
    //     for(int c = 0; c < m_data.channel(); ++c) {
    //         for(int h_high = 0; h_high < m_data.height(); ++h_high) {
    //             for(int w_high = 0; w_high < m_data.width(); ++w_high) {
    //                 m_data(n, c, h_high, w_high) = ave_pool::compute_window_ave(bottom_data, n, c, h_high, w_high, m_kernel_size);
    //             }
    //         }
    //     }
    // }
    wzp::ParallelRange(m_data.num(), [this, &bottom_data](int n) {
        for(int c = 0; c < m_data.channel(); ++c) {
            for(int h_high = 0; h_high < m_data.height(); ++h_high) {
                for(int w_high = 0; w_high < m_data.width(); ++w_high) {
                    m_data(n, c, h_high, w_high) = ave_pool::compute_window_ave(bottom_data, n, c, h_high, w_high, m_kernel_size);
                }
            }
        }
    });
    m_has_forwarded = true;
}


void AvePool::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_has_forwarded);
    check_backward(top);
    auto& top_grad = *(top[0]);
    // reshape the grad
    m_grad.Reshape(top_grad.num(), top_grad.channel(), top_grad.height() * m_kernel_size, top_grad.width() * m_kernel_size);
    // for(int n = 0; n < m_grad.num(); ++n) {
    //     for(int c = 0; c < m_grad.channel(); ++c) {
    //         for(int h_high = 0; h_high < top_grad.height(); ++h_high) {
    //             for(int w_high = 0; w_high < top_grad.width(); ++w_high) {
    //                 ave_pool::compute_window_grad(top_grad, n, c, h_high, w_high, m_kernel_size, m_grad);
    //             }
    //         }
    //     }
    // }
    wzp::ParallelRange(m_grad.num(), [this, &top_grad](int n) {
        for(int c = 0; c < m_grad.channel(); ++c) {
            for(int h_high = 0; h_high < top_grad.height(); ++h_high) {
                for(int w_high = 0; w_high < top_grad.width(); ++w_high) {
                    ave_pool::compute_window_grad(top_grad, n, c, h_high, w_high, m_kernel_size, m_grad);
                }
            }
        }
    });
    m_has_forwarded = false;
}


AvePool::AvePool(const int kernel_size) : PoolNode<F>(kernel_size), m_has_forwarded(false) {
    NeuronNode<F>::Resize();
}


} //nn

} //licon