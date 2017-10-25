#ifndef WZP_PATH_HPP
#define WZP_PATH_HPP

#include <string>
#include <vector>
#include <utility>


namespace wzp
{

class Path
{

public:
    /**
     * join the path, get the whole path
     * @param  paths [description]
     * @return       [description]
     */
    inline static std::string join(const std::vector<std::string>& paths) {
        //wash the strs
        std::vector<std::string> paths_(paths);
        for(auto& str : paths_) {
            if(str.back() == '/') str.pop_back();
        }
        std::string whole_path;
        for(size_t i = 0; i < paths_.size() - 1; ++i) {
            whole_path.append(paths_[i]);
            whole_path.push_back('/');
        }
        whole_path.append(paths_.back());
        return std::move(whole_path);
    }

    /**
     * get the base name of a path
     * @param  path [description]
     * @return      [description]
     */
    inline static std::string basename(const std::string& path) {
        std::string filename;
        size_t pos = path.find_last_of('/');
        if(pos != std::string::npos) filename = std::move(path.substr(pos+1));
        else filename = path;
        return std::move(filename);
    }


    /**
     * split the ext of a filename
     */
    inline static std::pair<std::string, std::string> splitext(const std::string& filename) {
        size_t pos = filename.find_last_of('.');
        return std::make_pair(filename.substr(0, pos), filename.substr(pos + 1));
    }

};


} //wzp



#endif //WZP_PATH_HPP