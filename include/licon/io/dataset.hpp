#ifndef LICON_IO_DATASET_HPP
#define LICON_IO_DATASET_HPP


#include <string>
#include <tuple>
#include <vector>


#include "licon/common.hpp"

#include "licon/utils/etensor.hpp"

/**
 * the basic interface of a dataset
 * will have two most important method to overlaod
 */

namespace licon
{

namespace io
{

template<typename... Args>
class Dataset {
public:
    //the construct will call by a string(input dir or input list or something)
    Dataset(std::string input) :m_input(std::move(input)) {}

    // de constrcut
    virtual ~Dataset() {}

    // the two need to overload function
    virtual std::tuple<Args...> GetItem(size_t index) = 0;

    virtual size_t Length() = 0;

    virtual std::tuple<int, int, int> Shape() = 0;

protected:
    std::string m_input;
    size_t m_length;

};


/**
 * the dataset of cifar10
 */
class Cifar10Dataset : public Dataset<std::vector<unsigned char>*, int> {
public:
    // the mode of train or test
    enum Mode { TRAIN, TEST};

public:
    Cifar10Dataset(std::string input, Mode load_mode = TRAIN);

    // the load interface
    void Load();

    std::tuple<std::vector<unsigned char>*, int> GetItem(size_t index) override;

    size_t Length() override ;

    std::tuple<int, int, int> Shape() override;

private:
    Mode m_mode;

    // the raw data of bin file
    std::vector<std::vector<unsigned char> > m_raw_data;

    // the raw label of bin file
    std::vector<int> m_raw_label;

    // private functions to load cifar10 data

};

/**
 * the dataset of mnist
 */
class MnistDataset : public Dataset<std::vector<unsigned char>*, int> {
public:
    enum Mode { TRAIN, TEST };
public:
    MnistDataset(std::string input, Mode load_mode = TRAIN);

    // load interface
    void Load();

    std::tuple<std::vector<unsigned char>*, int> GetItem(size_t index) override;

    size_t Length() override;

    std::tuple<int, int, int> Shape() override;

private:
    Mode m_mode;

    // the raw data of bin file
    std::vector<std::vector<unsigned char> > m_raw_data;

    // the raw label of bin file
    std::vector<int> m_raw_label;
};


/**
 * the loader of cifar10, an iterator will load data into -1 ~ 1
 */
class MnistCifar10Loader {
public:
    MnistCifar10Loader(Dataset<std::vector<unsigned char>*, int>* dataset, bool is_shuffle, int batch_size);

    // to set the new one
    void Register(Dataset<std::vector<unsigned char>*, int>* dataset, bool is_shuffle, int batch_size);

    bool has_next() const { return m_has_next; }

    // the next interface to get the batch data
    std::pair<utils::ETensor<F>, std::vector<int> > next();

private:
    bool m_has_next;

    Dataset<std::vector<unsigned char>*, int>* m_dataset;
    bool m_is_shuffle;
    int m_batch_size;

    // the indexs of batchs
    std::vector<size_t> m_indexs;
    size_t m_id; // the now index

    void Clear();

};

} //io

} //licon



#endif /*LICON_IO_DATASET_HPP*/