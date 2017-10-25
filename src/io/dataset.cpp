#include <cassert>

#include "licon/io/dataset.hpp"

#include <fstream>

#include "files/path.hpp"
#include "log/log.hpp"

#include "container/array_args.hpp"
#include "thread/parallel_algorithm.hpp"
#include "function/help_function.hpp"

using namespace std;

// some const numbers of cifar10
const static int CIFAR10_IMAGE_DEPTH = 3;
const static int CIFAR10_IMAGE_WIDTH = 32;
const static int CIFAR10_IMAGE_HEIGHT = 32;
const static int CIFAR10_IMAGE_AREA = CIFAR10_IMAGE_WIDTH * CIFAR10_IMAGE_HEIGHT;
const static int CIFAR10_IMAGE_SIZE = CIFAR10_IMAGE_DEPTH * CIFAR10_IMAGE_AREA;

// some const numbers of mnist
const static int MNIST_IMAGE_DEPTH = 1;
const static int MNIST_IMAGE_WIDTH = 28;
const static int MNIST_IMAGE_HEIGHT = 28;
const static int MNIST_IMAGE_AREA = MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT;
const static int MNIST_IMAGE_SIZE = MNIST_IMAGE_DEPTH * MNIST_IMAGE_AREA;

namespace licon
{

namespace io
{

namespace cifar10
{

void parse_cifar10(const std::string& input, vector<vector<unsigned char> >* datas, vector<int>* labels) {
    std::ifstream ifs(input, std::ios::in | std::ios::binary);
    ASSERT(ifs, "can not open cifar10 file", input);
    unsigned char label;
    vector<unsigned char> buf(CIFAR10_IMAGE_SIZE);
    while(ifs.read(reinterpret_cast<char *>(&label), 1)) {
        if (!ifs.read(reinterpret_cast<char *>(&buf[0]), CIFAR10_IMAGE_SIZE)) break;
        datas->emplace_back(buf);
        labels->emplace_back(static_cast<int>(label));
    }
    ifs.close();
}

} //details of cifar10

namespace mnist
{

void parse_mnist_images(const std::string& input, vector<vector<unsigned char> >* datas) {
    std::ifstream ifs(input, std::ios::in | std::ios::binary);
    ASSERT(ifs, "can not open mnist file", input);
    int magic;
    ifs.read(reinterpret_cast<char*>(&magic), 4); //magic
    ifs.read(reinterpret_cast<char*>(&magic), 4); //num
    ifs.read(reinterpret_cast<char*>(&magic), 4); //row
    ifs.read(reinterpret_cast<char*>(&magic), 4); //col
    vector<unsigned char> buf(MNIST_IMAGE_SIZE);
    while(ifs.read(reinterpret_cast<char*>(&buf[0]), MNIST_IMAGE_SIZE)) {
        datas->emplace_back(buf);
    }
    ifs.close();
}

void parse_mnist_labels(const std::string& input, vector<int>* labels) {
    std::ifstream ifs(input, std::ios::in | std::ios::binary);
    ASSERT(ifs, "can not open mnist file", input);
    int magic;
    ifs.read(reinterpret_cast<char*>(&magic), 4); //magic
    ifs.read(reinterpret_cast<char*>(&magic), 4); //num
    unsigned char label;
    while(ifs.read(reinterpret_cast<char*>(&label), 1)) {
        labels->emplace_back(static_cast<int>(label));
    }
    ifs.close();
}

} //details of mnist load



/**
 * cifar10 functions
 */
Cifar10Dataset::Cifar10Dataset(std::string input, Mode load_mode)
    : Dataset<vector<unsigned char>*, int>(std::move(input)), m_mode(load_mode)
{}

void Cifar10Dataset::Load() {
    if(m_mode == TRAIN) {
        cifar10::parse_cifar10(wzp::Path::join({m_input, "data_batch_1.bin"}), &m_raw_data, &m_raw_label);
        cifar10::parse_cifar10(wzp::Path::join({m_input, "data_batch_2.bin"}), &m_raw_data, &m_raw_label);
        cifar10::parse_cifar10(wzp::Path::join({m_input, "data_batch_3.bin"}), &m_raw_data, &m_raw_label);
        cifar10::parse_cifar10(wzp::Path::join({m_input, "data_batch_4.bin"}), &m_raw_data, &m_raw_label);
        cifar10::parse_cifar10(wzp::Path::join({m_input, "data_batch_5.bin"}), &m_raw_data, &m_raw_label);
    } else {
        cifar10::parse_cifar10(wzp::Path::join({m_input, "test_batch.bin"}), &m_raw_data, &m_raw_label);
    }
    wzp::log::info("Cifar10 DataSet Loaded...");
}


std::tuple<vector<unsigned char>*, int> Cifar10Dataset::GetItem(size_t index) {
    return std::make_tuple(&m_raw_data[index], m_raw_label[index]);
}

size_t Cifar10Dataset::Length() {
    return m_raw_label.size();
}

std::tuple<int, int, int> Cifar10Dataset::Shape() {
    return std::make_tuple(CIFAR10_IMAGE_DEPTH, CIFAR10_IMAGE_HEIGHT, CIFAR10_IMAGE_WIDTH);
}

MnistDataset::MnistDataset(std::string input, Mode load_mode)
    : Dataset<vector<unsigned char>*, int>(std::move(input)), m_mode(load_mode)
{}

void MnistDataset::Load() {
    if(m_mode == TRAIN) {
        mnist::parse_mnist_images(wzp::Path::join({m_input, "train-images-idx3-ubyte"}), &m_raw_data);
        mnist::parse_mnist_labels(wzp::Path::join({m_input, "train-labels-idx1-ubyte"}), &m_raw_label);
    } else {
        mnist::parse_mnist_images(wzp::Path::join({m_input, "t10k-images-idx3-ubyte"}), &m_raw_data);
        mnist::parse_mnist_labels(wzp::Path::join({m_input, "t10k-labels-idx1-ubyte"}), &m_raw_label);
    }
    wzp::log::info("Mnist DataSet Loaded...");
}


std::tuple<vector<unsigned char>*, int> MnistDataset::GetItem(size_t index) {
    return std::make_tuple(&m_raw_data[index], m_raw_label[index]);
}

size_t MnistDataset::Length() {
    return m_raw_label.size();
}


std::tuple<int, int, int> MnistDataset::Shape() {
    return std::make_tuple(MNIST_IMAGE_DEPTH, MNIST_IMAGE_HEIGHT, MNIST_IMAGE_WIDTH);
}

/**
 * ######################################
 */

/**
 * the data loader part
 */

MnistCifar10Loader::MnistCifar10Loader(Dataset<std::vector<unsigned char>*, int>* dataset, bool is_shuffle, int batch_size)
    : m_has_next(true), m_dataset(dataset), m_is_shuffle(is_shuffle), m_batch_size(batch_size)
{
    Clear();
}

void MnistCifar10Loader::Register(Dataset<std::vector<unsigned char>*, int>* dataset, bool is_shuffle, int batch_size) {
    m_dataset = dataset;
    m_is_shuffle = is_shuffle;
    m_batch_size = batch_size;
    Clear();
}

void MnistCifar10Loader::Clear() {
    m_id = 0;
    m_indexs.clear();
    m_indexs.reserve(m_dataset->Length());
    for(size_t i = 0; i < m_dataset->Length(); ++i) m_indexs.emplace_back(i);
    if(m_dataset->Length()) m_has_next = true;
    else m_has_next = false;
    if(m_is_shuffle) {
        wzp::array_args<size_t>::shuffle(m_indexs);
    }
}

std::pair<utils::ETensor<F>, std::vector<int> > MnistCifar10Loader::next() {
    assert(m_has_next);
    using utils::ETensor;
    auto batch_size = std::min(static_cast<size_t>(m_batch_size), m_dataset->Length() - m_id);
    auto shapes = m_dataset->Shape();
    ETensor<F> batch_data(batch_size, GET_0(shapes), GET_1(shapes), GET_2(shapes));
    vector<int> labels(batch_size);
    wzp::ParallelRange(batch_size,
        [this, &batch_data, &labels](size_t i) {
            auto offset = m_id + i;
            auto data_label = m_dataset->GetItem(m_indexs[offset]);
            auto data_ptr = GET_0(data_label);
            int label = GET_1(data_label);
            F* tensor_ptr = batch_data.mutable_ptr(i);
            for(size_t j = 0; j < data_ptr->size(); ++j) {
                F tmp = static_cast<F>((*data_ptr)[j]);
                *(tensor_ptr + j) = -1.0 + 2.0 * (tmp / 255.);
            }
            labels[i] = label;
    });
    m_id += batch_size;
    if(m_id >= m_dataset->Length()) m_has_next = false;
    return std::make_pair(std::move(batch_data), std::move(labels));
}

} //io

} //licon