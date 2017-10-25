#ifndef LICON_NN_MAX_POOL_HPP
#define LICON_NN_MAX_POOL_HPP

#include "licon/nn/node/pool.hpp"

namespace licon
{

namespace nn
{

class MaxPool : public PoolNode<F> {

public:
    static std::unique_ptr<OpNode<F> > CreateMaxPool(const int kernel_size);

    // the name
    virtual inline std::string name() const { return "max_pool"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);

protected:
    // need a mask to save the max area
    utils::ETensor<char> m_mask; //when forward save the max location, use char to save the memory
    bool m_has_forwarded;


private:
    MaxPool(const int kernel_size);

};

} //nn

} //licon

#endif /*LICON_NN_MAX_POOL_HPP*/