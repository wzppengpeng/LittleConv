#ifndef FILE_LOADER_HPP_
#define FILE_LOADER_HPP_


#include <fstream>
#include <string>
#include <vector>
#include <functional>

namespace wzp
{

template<typename R>
class FileLoader
{

public:
    FileLoader() : m_can_iter(true) {}

    //the regis function, to regis a function object to get process a line
    template<typename Fun>
    void regis(Fun f, const std::string& filename) {
        m_fn = std::move(f);
        ifile.open(filename, std::ios::in);
        if(!ifile) m_can_iter = false;
        else m_can_iter = true;
    }

    //check if has next
    bool has_next() {
        if(!m_can_iter) return false;
        if(std::getline(ifile, m_buffer)) {}
        else {
            m_can_iter = false;
            ifile.close();
        }
        return m_can_iter;
    }

    //get the next line and process to R
    R next() {
        R r = m_fn(m_buffer);
        if(ifile.eof()) {
            m_can_iter = false;
            ifile.close();
        }
        return std::move(r);
    }

private:
    //the function object
    std::function<R(const std::string&)> m_fn;

    //can iter sysbol
    bool m_can_iter;

    //the file stream
    std::ifstream ifile;
    std::string m_buffer;
};


/**
 * STL iterator type file loader
 */
//use just like STL
//return an iterator point return data
template<typename R>
class FileLoad
{
    class FileIter;
    typedef long long ll;

public:
    FileLoad() : m_can_iter(false)
    {}

    //the regis function, to regis a function object to get process a line
    template<typename Fun>
    void regis(Fun f, const std::string& filename) {
        m_fn = std::move(f);
        ifile.open(filename, std::ios::in);
        if(!ifile) m_can_iter = false;
        else m_can_iter = true;
    }

    //the begin
    FileIter begin() {
        //load first line
        if(!m_can_iter) {
            return FileIter(-1, *this);
        }
        if(std::getline(ifile, m_buffer)) {
            return FileIter(0, *this);
        }
        else {
            ifile.close();
            return FileIter(-1, *this);
        }
    }

    //the end
    FileIter end() {
        return FileIter(-1, *this);
    }

private:
    bool m_can_iter;

    //the function object
    std::function<R(const std::string&)> m_fn;

    //the file stream
    std::ifstream ifile;
    std::string m_buffer;

    R m_r;

private:

    class FileIter
    {
    public:
        //constructor
        FileIter(ll idx, FileLoad& load) : m_idx(idx), m_load(load)
        {
        }

        //dereferece
        R& operator* () {
            m_load.m_r = std::move(m_load.m_fn(m_load.m_buffer));
            return m_load.m_r;
        }

        //self increament
        FileIter& operator++ () {
            //read next line
            if(std::getline(m_load.ifile, m_load.m_buffer)) {
                ++m_idx;
            }
            else {
                m_load.m_can_iter = false;
                m_idx = -1;
                m_load.ifile.close();
            }
            return *this;
        }

        //==
        bool operator== (const FileIter& other) {
            return other.m_idx == this->m_idx;
        }

        bool operator!= (const FileIter& other) {
            return other.m_idx != this->m_idx;
        }

    private:
        ll m_idx;
        FileLoad& m_load;
    };
};


} //wzp

#endif //FILE_LOADER_HPP_