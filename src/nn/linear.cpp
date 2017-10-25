#include "licon/nn/node/linear.hpp"

#include "licon/nn/init.hpp"


#include "licon/utils/ematrix_args.hpp"

#include "licon/utils/transformer.hpp"

#include "function/help_function.hpp"

using namespace std;

const static F WEIGHT_MEAN = 0.;
const static F WEIGHT_VAR = 0.02;
const static F BIAS_VAL = 0.;


namespace licon
{

namespace nn
{

std::unordered_map<std::string, utils::ETensor<F>* > Linear::StateDict() {
    CHECK(m_node_name.empty() == false);
    std::unordered_map<std::string, utils::ETensor<F>* > state_dict_linear;
    state_dict_linear.emplace(m_node_name + "W", weight());
    state_dict_linear.emplace(m_node_name + "B", bias());
    return state_dict_linear;
}

void Linear::LoadStateDict(const std::unordered_map<std::string, utils::ETensor<F> >& state_dict) {
    CHECK(m_node_name.empty() == false);
    ASSERT(state_dict.find(m_node_name + "W") != state_dict.end(), "Miss Fc Weight", m_node_name);
    ASSERT(state_dict.find(m_node_name + "B") != state_dict.end(), "Miss Fc Bias", m_node_name);
    m_weights = state_dict.find(m_node_name + "W")->second;
    m_bias = state_dict.find(m_node_name + "B")->second;
}

std::unique_ptr<OpNode<F> > Linear::CreateLinear(int input, int output) {
    return std::unique_ptr<OpNode<F> >(new Linear(input, output));
}

void Linear::Forward(const std::vector<utils::ETensor<F>* >& bottom) {
    ASSERT(bottom[0]->count(1) == static_cast<size_t>(m_input_dim), "The Shape MisMatch", m_input_dim);
    check_forward(bottom);
    // reshape the output data
    m_bottom_data = bottom[0];
    m_data.Reshape(m_bottom_data->num(), m_output_dim, 1, 1);
    auto view_matrix = m_bottom_data->ViewInPlace<wzp::EMatrix<F> >();
    // matrix dot
    auto weight_matrix = m_weights.ViewInPlace<wzp::EMatrix<F> >();
    auto bias_matrix = m_bias.ViewInPlace<wzp::EMatrix<F> >();
    auto res_data = view_matrix * weight_matrix + bias_matrix;
    utils::transform_matrix_to_tensor(res_data, m_data);
}

void Linear::Backward(const std::vector<utils::ETensor<F>* >& top) {
    assert(m_bottom_data != nullptr);
    check_backward(top);
    ASSERT(top[0]->num() == m_data.num() && top[0]->channel() == m_data.channel() && top[0]->height() == 1
        && top[0]->width() == 1, "Shape MisMatch", m_output_dim);
    m_grad.Reshape(m_bottom_data->shape()); //reshape the grad into the input shape
    auto& top_grad = *top[0];
    auto grad_matrix = top_grad.ViewInPlace<wzp::EMatrix<F> >();
    auto weight_matrix = m_weights.ViewInPlace<wzp::EMatrix<F> >();
    auto grad_res = grad_matrix * (weight_matrix.transpose()); //n * d
    utils::transform_matrix_to_tensor(grad_res, m_grad);

    // compute the backward grad of weights and bias
    auto view_matrix = m_bottom_data->ViewInPlace<wzp::EMatrix<F> >();
    auto weights_grad_maxtrix = std::move(view_matrix.transpose() * grad_matrix);
    auto bias_grad_matrix = std::move(wzp::EMatrix<F>(1, view_matrix.rows(), 1) * grad_matrix);
    // transform matrix to tensor
    utils::transform_matrix_to_tensor(weights_grad_maxtrix, m_weights_grad);
    utils::transform_matrix_to_tensor(bias_grad_matrix, m_bias_grad);
    m_bottom_data = nullptr;
}

void Linear::InitParameters() {
    if(m_has_initialzed == false) {
        // nn::ParamInit<F>::normal(m_weights, WEIGHT_MEAN, WEIGHT_VAR);
        nn::ParamInit<F>::xavier_uniform(m_weights);
        nn::ParamInit<F>::constant(m_bias, BIAS_VAL);
        m_has_initialzed = true;
    }
}

std::vector<std::pair<utils::ETensor<F>*, utils::ETensor<F>* > > Linear::RegisterWeights() {
    return {{&m_weights, &m_weights_grad}, {&m_bias, &m_bias_grad}};
}

void Linear::InitWeights(F mean, F var) {
    nn::ParamInit<F>::normal(m_weights, mean, var);
}

void Linear::InitBias(F val) {
    nn::ParamInit<F>::constant(m_bias, val);
}

Linear::Linear(int input, int output)
    : NeuronNode<F>(),
    m_bottom_data(nullptr),
    m_input_dim(input),
    m_output_dim(output),
    m_has_initialzed(false),
    m_weights(input, output, 1, 1),
    m_bias(1, output, 1, 1),
    m_weights_grad(input, output, 1, 1),
    m_bias_grad(1, output, 1, 1)
{
    NeuronNode<F>::Resize();
    // init params
    InitParameters();
}


} //nn

} //licon