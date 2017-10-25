#include "licon/utils/evaluation.hpp"

#include "licon/io/dataset.hpp"
#include "licon/nn/operation_node.hpp"

#include "function/help_function.hpp"

using namespace std;

namespace licon
{

namespace utils
{

std::unique_ptr<Evaluation<std::vector<unsigned char>*, int> > Accuracy::CreateAccuracy(
            std::unique_ptr<nn::OpNode<F> >& model, io::Dataset<std::vector<unsigned char>*, int>& eval_dataset) {
    return std::unique_ptr<Evaluation<vector<unsigned char>*, int> > (new Accuracy(model, eval_dataset));
}


Accuracy::Accuracy(std::unique_ptr<nn::OpNode<F> >& model, io::Dataset<std::vector<unsigned char>*, int>& eval_dataset)
    : Evaluation<vector<unsigned char>*, int>(model, eval_dataset) {}

F Accuracy::Run() {
    auto find_max_indexs = [](const utils::ETensor<F>& res) {
        vector<int> max_indics(res.num(), 0);
        for(int i = 0; i < res.num(); ++i) {
            int max_j = 0;
            F max_sco = 0.0;
            for(int j = 0; j < res.channel(); ++j) {
                if(res(i, j) > max_sco) {
                    max_j = j;
                    max_sco = res(i, j);
                }
            }
            max_indics[i] = max_j;
        }
        return std::move(max_indics);
    };
    io::MnistCifar10Loader loader(&m_eval_dataset, false, 512);
    int accurate_num = 0, total = 0;
    while(loader.has_next()) {
        auto data_label = loader.next();
        utils::ETensor<F>* input = &data_label.first;
        // model forward
        m_model->Forward({input});
        auto max_indics = find_max_indexs(*m_model->data());
        for(size_t i = 0; i < max_indics.size(); ++i) {
            if(max_indics[i] == data_label.second[i]) ++accurate_num;
            ++total;
        }
    }
    return static_cast<F>(accurate_num) / static_cast<F>(total);
}

} //utils

} //licon