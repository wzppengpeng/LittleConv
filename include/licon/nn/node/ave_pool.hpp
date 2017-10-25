#ifndef LICON_NN_AVE_POOL_HPP_
#define LICON_NN_AVE_POOL_HPP_

#include "licon/nn/node/pool.hpp"

namespace licon
{

namespace nn
{

class AvePool : public PoolNode<F> {
public:
    // the heap create functuin
    static std::unique_ptr<OpNode<F> > CreateAvePool(const int kernel_size);

    // the name of thisnode
    virtual inline std::string name() const { return "ave_pool"; }

    // the forward and backward
    virtual void Forward(const std::vector<utils::ETensor<F>* >& bottom);

    virtual void Backward(const std::vector<utils::ETensor<F>* >& top);


protected:
    bool m_has_forwarded;

private:
    AvePool(const int kernel_size);

};

} //nn

} //licon

#endif /*LICON_NN_AVE_POOL_HPP_*/