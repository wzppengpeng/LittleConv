#ifndef MULTITON_HPP
#define MULTITON_HPP

#include <string>
#include <unordered_map>
#include <memory>

namespace wzp {

template<typename T, typename K = std::string>
class Multiton
{
public:
    template<typename... Args>
    static std::shared_ptr<T> Instance(const K& key, Args&&... args) {
        return GetInstance(key, std::forward<Args>(args)...);
    }

    template<typename... Args>
    static std::shared_ptr<T> Instance(K&& key, Args&&... args) {
        return GetInstance(key, std::forward<Args>(args)...);
    }

private:
    Multiton(void);
    virtual ~Multiton() {}
    Multiton(const Multiton&) = delete;
    Multiton(Multiton&&) = delete;
    Multiton& operator = (const Multiton& ) = delete;
    Multiton& operator = (Multiton&&) = delete;

    //the get instance function
    template<typename Key, typename... Args>
    static std::shared_ptr<T> GetInstance(Key&& key, Args&&... args) {
        std::shared_ptr<T> instance(nullptr);
        auto it = m_map.find(std::forward<Key>(key));
        if(it != m_map.end()) {
            instance = it->second;
        }
        else {
            instance = std::make_shared<T>(std::forward<Args>(args)...);
            m_map.emplace(std::forward<Key>(key), instance);
        }
        return instance;
    }

private:
    static std::unordered_map<K, std::shared_ptr<T>> m_map;
};

template<typename T, typename K>
std::unordered_map<K, std::shared_ptr<T>> Multiton<T, K>::m_map;

} // wzp


#endif // MULTITON_HPP
