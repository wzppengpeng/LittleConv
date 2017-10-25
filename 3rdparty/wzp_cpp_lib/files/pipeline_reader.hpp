#ifndef PIPELINE_READER_HPP
#define PIPELINE_READER_HPP

#include <fstream>
#include <string>
// #include <iostream>
#include <atomic>
// #include "container/threadsafe_queue.h"

#include "thread/taskjob.hpp"

namespace wzp {

class PiplelineReader {
public:
    /*read data from a file and process by line
    that is to say one thread read data from file, one thread process data
    to avoid the mutex problem
    */
    //this function is to handle the ios::in type
    static bool read_process(const std::string& filename,
     const std::function<void(const std::string&)>& process_fun) {
        std::ifstream infile(filename, std::ios::in);
        if(!infile) return false;
        //read first line
        std::string process_line;
        std::string read_line;
        std::ifstream* infile_ptr = &infile;
        std::getline(*infile_ptr, process_line);
        bool done(false);
        while(!done) {
            Task<void()> read_task([infile_ptr, &read_line, &done]{
                if(!std::getline(*infile_ptr, read_line)) {
                    done = true;
                }
            });
            auto fut = read_task.run();
            process_fun(process_line);
            fut.get();
            process_line = std::move(read_line);
        }
        infile.close();
        return true;
    }
};

} // wzp

#endif // PIPELINE_READER_HPP
