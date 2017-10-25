#ifndef GRID_HPP
#define GRID_HPP

#include <tuple>
#include <vector>
#include <string>

#include <fstream>
#include <iostream>

#include <stdexcept>

#include "my_string/string.hpp"


namespace wzp
{

/*the tuple walk to change data into string*/
namespace details {
template<typename Tuple, size_t N>
struct TupleWriter
{
    static void write(const Tuple& t, std::ostream* o_ptr) {
        TupleWriter<Tuple, N - 1>::write(t, o_ptr);
        *o_ptr<<','<<std::get<N - 1>(t);
    }
};

//the paricial type
template<typename Tuple>
struct TupleWriter<Tuple, 1>
{
    static void write(const Tuple& t, std::ostream* o_ptr) {
        *o_ptr<<std::get<0>(t);
    }
};

template<typename... Args>
void write(const std::tuple<Args...>& t, std::ostream* o_ptr) {
    TupleWriter<decltype(t), sizeof...(Args)>::write(t, o_ptr);
}
/*--------------------------------------------*/

/*the tuple reader read data into a tuple*/
template<typename Tuple, size_t N>
struct TupleReader
{
    static void read(Tuple& t, const std::vector<std::string>& strs) {
        TupleReader<Tuple, N - 1>::read(t, strs);
        auto temp = convert_string<typename std::tuple_element<N - 1, Tuple>::type>(strs[N - 1]);
        std::get<N - 1>(t) = std::move(temp);
    }
};

//the parcial type
template<typename Tuple>
struct TupleReader<Tuple, 1>
{
    static void read(Tuple& t, const std::vector<std::string>& strs) {
        auto temp = convert_string<typename std::tuple_element<0, Tuple>::type>(strs[0]);
        std::get<0>(t) = std::move(temp);
    }
};

template<typename... Args>
void read(std::tuple<Args...>& t, const std::vector<std::string>& strs) {
    TupleReader<typename std::remove_reference<decltype(t)>::type, sizeof...(Args)>::read(t, strs);
}

template<typename Tuple, size_t N>
struct TuplePrinter
{
    static void print(const Tuple& tuple_) {
        TuplePrinter<Tuple, N - 1>::print(tuple_);
        std::cout<<" "<<std::get<N - 1>(tuple_);
    }
};

template<typename Tuple>
struct TuplePrinter<Tuple, 1>
{
    static void print(const Tuple& tuple_) {
        std::cout<<std::get<0>(tuple_);
    }
};

template<typename... Args>
void print_tuple(const std::tuple<Args...>& tuple_) {
    std::cout<<"( ";
    TuplePrinter<decltype(tuple_), sizeof...(Args)>::print(tuple_);
    std::cout<<" )"<<std::endl;
}

} //details
/*====================================*/

/************************************/
template<size_t N, typename... Args>
class Grid {
private:
    std::vector<std::tuple<Args...>> m_grid;//meta data stored here
    enum { ColNum = sizeof...(Args)};

private:
    inline std::tuple<Args...>& get_mutable_row_at(size_t x) { return m_grid[x]; }

public:
    //get the rows
    Grid() : m_grid(N) {}

    //cols rows and the vector capacity
    /**
     * get the cols
     * @return the cols number
     */
    inline size_t cols() const { return ColNum; }

    inline size_t rows() const { return m_grid.size(); }

    inline size_t capacity() const { return m_grid.capacity(); }


    //allocate memory, just as the same as vector
    /**
     * [resize description]
     * @param s [new size]
     */
    inline void resize(size_t s) { m_grid.resize(s); }

    /**
     * reserve the vector
     * @param s the new size of vector
     */
    inline void reserve(size_t s) { m_grid.reserve(s); }

    //get the row, return a tuple, can use std::tie and std::get<N> to get data

    inline const std::tuple<Args...>& get_row_at(size_t x) const { return m_grid[x]; }

    //get the data
    //get the data by template const static number at<1, 2>
    template<size_t r, size_t c>
    inline auto at()
    -> decltype(std::get<c>(this->get_mutable_row_at(r))) {
        return std::get<c>(this->get_mutable_row_at(r));
    }

    //change the data
    //just changge the row tuple, the pakage tuple
    template<typename... Param>
    void reset_row(size_t x, Param&&... params) {
        get_mutable_row_at(x) = std::move(std::make_tuple(std::forward<Param>(params)...));
    }

    //push back
    //the pakage of emplace_back
    template<typename... Param>
    void emplace_back(Param&&... params) {
        m_grid.emplace_back(std::forward<Param>(params)...);
    }

    //emplace_back a tuple into the vector
    /**
     * [push_back a tuple into the vector]
     * @param t [a new tuple]
     */
    void push_back(std::tuple<Args...>&& t) { m_grid.emplace_back(t); }
    void push_back(const std::tuple<Args...>& t) { m_grid.push_back(t); }

    //pop back and clear
    inline void pop_back() { m_grid.pop_back(); }
    inline void clear() { m_grid.clear(); }

    //some data interactive
    //first change the grid into csv file
    /**
     * change the grid into csv file
     * @param filename the filename to store
     */
    void to_csv(const char* filename) noexcept {
        std::ofstream ofile(filename, std::ios::out);
        auto o_ptr = &ofile;
        for(size_t i = 0; i < rows(); ++i) {
            details::write(get_row_at(i), o_ptr);
            *o_ptr<<std::endl;
        }
        ofile.close();
    }

    //read data from a csv file
    /**
     * read csv file then put data into a grid
     * @param filename the csv filename
     */
    void from_csv(const char* filename) noexcept {
        //first clean the grid
        clear();
        std::ifstream ifile(filename, std::ios::in);
        std::string line;
        while(std::getline(ifile, line)) {
            auto strs = split_string(line, ',');
            std::tuple<Args...> new_row;
            details::read(new_row, strs);
            m_grid.emplace_back(std::move(new_row));
        }
        ifile.close();
    }

    //print this grid
    /**
     * print this grid
     */
    void print_grid() {
        for(size_t i = 0; i < rows(); ++i) {
            details::print_tuple(get_row_at(i));
        }
    }

};

/******************************/

/*here is some convert help functions*/

/*********************************/

} //wzp

#endif /*GRID_HPP*/