#ifndef FILES_HPP_
#define FILES_HPP_

#include <vector>
#include <string>
#include <dirent.h>

namespace wzp{
    class file_system{
    public:
        /*get all files info*/
        inline static bool get_all_files(const std::string& dir_path,
                    std::vector<std::string>& files_vec){
            struct dirent* ptr;
            DIR* dir;
            dir = opendir(dir_path.c_str());
            if(!dir) return false;
            while((ptr = readdir(dir))){
                if(ptr->d_name[0] == '.') continue;
                files_vec.emplace_back(ptr->d_name);
            }
            closedir(dir);
            return true;
        }

        /*get the dir files by the postfix*/
        inline static bool get_files_by_postfix(const std::string& dir_path,
                    std::vector<std::string>& files_vec, const std::string& postfix){
            struct dirent* ptr;
            DIR* dir;
            dir = opendir(dir_path.c_str());
            if(!dir) return false;
            while((ptr = readdir(dir))){
                if(ptr->d_name[0] == '.') continue;
                std::string filename(ptr->d_name);
                if(filename.find(std::move("." + postfix)) != std::string::npos){
                    files_vec.emplace_back(filename);
                }
            }
            closedir(dir);
            return true;
        }

        // clear dir
        inline static void clear_dir(const std::string& dir_path) {
            std::string cmd("mkdir -p " + dir_path);
            int res = system(cmd.c_str());
            cmd = "rm -rf " + dir_path + "/*";
            res = system(cmd.c_str());
        }
    };
}

#endif /*FILES_HPP_*/