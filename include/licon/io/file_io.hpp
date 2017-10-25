#ifndef LICON_FILE_IO_HELPER_HPP
#define LICON_FILE_IO_HELPER_HPP



#include <string>
#include <fstream>

#include "log/log.hpp"

#include "licon/nn/operation_node.hpp"


namespace licon
{

namespace io
{

// the buffer type
typedef std::string Buffer;

// the function to write buffer to disk
inline static void write_buffer_to_disk(const Buffer& buffer, const std::string& filename) {
    std::ofstream ofile(filename, std::ios::binary);
    ofile.write(buffer.c_str(), sizeof(char) * buffer.size());
    ofile.close();
}

// the function to read buffer to disk
inline static Buffer read_buffer_from_disk(const std::string& filename) {
    std::ifstream ifile;
    ifile.open(filename);
    ASSERT(ifile, "the file can not open", filename);
    Buffer buffer((std::istreambuf_iterator<char>(ifile)),
        std::istreambuf_iterator<char>());
    ifile.close();
    return buffer;
}

/**
 * the saver to save model
 */
class Saver {

public:
    // static function to save the operation node
    static void Save(const std::string& filename, nn::Model& model);

    // static function to load the operation node
    static void Load(const std::string& filename, nn::Model* model);
};

} //io

} //licon

#endif /*LICON_FILE_IO_HELPER_HPP*/