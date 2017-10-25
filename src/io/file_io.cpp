#include "licon/io/file_io.hpp"

#include "licon/common.hpp"

#include "licon/utils/etensor.hpp"
#include "licon/utils/serialize.hpp"

#include "function/help_function.hpp"

using namespace std;

namespace licon
{

namespace io
{

void Saver::Save(const std::string& filename, nn::Model& model) {
    // check the model name
    ASSERT(model->node_name().empty() == false, "The Model Need A Name");
    // get the state dict
    auto state_dict = model->StateDict();
    Buffer buffer;
    // add to buffer
    // add the model name
    utils::Serializer::serialize(model->node_name(), &buffer);
    // add the nodename and its weights and bias
    for(auto& p : state_dict) {
        utils::Serializer::serialize(std::string(p.first), &buffer);
        utils::Serializer::serialize(*(p.second), &buffer);
    }
    // save to disk
    write_buffer_to_disk(buffer, filename);
}

void Saver::Load(const std::string& filename, nn::Model* model) {
    // load the disk data to buffer
    auto buffer = read_buffer_from_disk(filename);
    // first get the model name
    std::string model_name;
    auto offset = utils::Serializer::deserialize(buffer, 0, model_name);
    // set the model
    (*model)->set_node_name(model_name);
    // read the buffer into tensors
    unordered_map<string, utils::ETensor<F> > state_dict;
    while(offset < buffer.size()) {
        // get the state name
        std::string tmp_name;
        utils::ETensor<F> weight;
        offset =  utils::Serializer::deserialize(buffer, offset, tmp_name);
        offset = utils::Serializer::deserialize(buffer, offset, weight);
        state_dict.emplace(std::move(tmp_name), std::move(weight));
    }
    // load the state dict
    (*model)->LoadStateDict(state_dict);
}

} //io

} //licon