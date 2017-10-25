#ifndef GRID_PLUS_HPP_
#define GRID_PLUS_HPP_
#include <cassert>
#include <sstream>

#include <leveldb/db.h> // only this machine have leveldb can use this
// #include <mysql.h> // only this machin have mysql can use this

#include "container/grid.hpp"


using std::string;

namespace wzp
{

//here will add some convert functions
/*convert normal Grid which can change into the stringstream by '<<' */


template<size_t N, typename... Args>
void SaveNormalGridToLeveldb(const std::string& grid_name,
 leveldb::DB* db, const Grid<N, Args...>& grid) {
    if(db == nullptr) throw std::logic_error("the db is not open");
    leveldb::Status status;
    //now save data's
    std::ostringstream ostr_data;
    auto o_ptr = &ostr_data;
    for(size_t i = 0; i < grid.rows(); ++i) {
        details::write(grid.get_row_at(i), o_ptr);
        *o_ptr<<'\n';
    }
    status = db->Put(leveldb::WriteOptions(), grid_name, ostr_data.str());
    assert(status.ok());
}

template<size_t N, typename... Args>
void ReadNormalGridFromLeveldb(const std::string& grid_name,
 leveldb::DB* db, Grid<N, Args...>& grid) noexcept {
    leveldb::Status status;
    std::string cache;
    status = db->Get(leveldb::ReadOptions(), grid_name, &cache);
    auto cache_vec = wzp::split_string(cache, '\n');
    for(const auto& str : cache_vec) {
        auto str_vec = wzp::split_string(str, ',');
        std::tuple<Args...> new_row;
        details::read(new_row, str_vec);
        grid.push_back(std::move(new_row));
    }
}

/*********************************************************************/

} //wzp

#endif /*GRID_PLUS_HPP_*/