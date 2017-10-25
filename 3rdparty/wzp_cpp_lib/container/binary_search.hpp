#ifndef BINARY_SEARCH_HPP
#define BINARY_SEARCH_HPP

/**
 * use binary search to generate two container bin_set and bin_map
 */
#include <algorithm>
#include <vector>
#include <utility>

namespace wzp {

/**
 * the set, its efficiency is very nice
 */
template<typename T>
class BinSet {
public:
    using Iter = typename std::vector<T>::iterator;

    BinSet() = default;

    //construct by vector
    BinSet(const std::vector<T>& data) : v_(data) {
        init();
    }

    BinSet(std::vector<T>&& data) : v_(data) {
        init();
    }

    //if not get data by construct, data can be create
    /**
     * create the vector data
     * @param data the raw data
     */
    void create(const std::vector<T>& data) {
        v_ = data;
        init();
    }

    void create(std::vector<T>&& data) {
        v_ = data;
        init();
    }

    //find and count function
    bool exist(T&& t) const {
        return std::binary_search(std::begin(v_), std::end(v_), std::forward<T>(t));
    }

    //return a iter
    Iter find(T&& t) {
        auto it = std::lower_bound(std::begin(v_), std::end(v_), std::forward<T>(t));
        if(it == v_.end() || *it < t) {
            return v_.end();
        }
        else {
            return it;
        }
    }

    std::pair<Iter, Iter> range(T&& t) {
        auto p = std::equal_range(std::begin(v_), std::end(v_), std::forward<T>(t));
        return std::move(p);
    }

    int count(T&& t) {
        auto p = range(std::forward<T>(t));
        return std::distance(p.first, p.second);
    }

    //insert and erase function, but very very low effective
    void insert(T&& t) {
        auto it = std::lower_bound(std::begin(v_), std::end(v_), std::forward<T>(t));
        v_.insert(it, std::forward<T>(t));
    }

    inline void erase(Iter it) {
        v_.erase(it);
    }

    void erase(T&& t) {
        auto it = find(std::forward<T>(t));
        erase(it);
    }


private:
    std::vector<T> v_; // the container

private:
    inline void init() {
        std::sort(std::begin(v_), std::end(v_)); // only can sort from low to up
    }

};

//the BinMap should be vector<pair<Key, Value>> Type
template<typename Key, typename Value>
class BinMap {
public:
    using MetaData = typename std::vector<std::pair<Key, Value>>;
    using T = typename std::pair<Key, Value>;
    using Iter = typename std::vector<std::pair<Key, Value>>::iterator;

    /**
     * Constrctors
     */
    BinMap() = default;

    /**
     * the Constructor by vector<pair<Key, Value>>
     */
    BinMap(const MetaData& data) : v_(data) {
        init();
    }

    BinMap(MetaData&& data) : v_(data) {
        init();
    }

    /**
     * create the meta data by another vector<pair>
     * @param data the normal vector data
     */
    void create(const MetaData& data) {
        v_ = data;
        init();
    }

    void create(MetaData&& data) {
        v_  = data;
        init();
    }

    //find and count function
    bool exist(const Key& t) const {
        return std::binary_search(std::begin(v_), std::end(v_), T(t, Value()), [](const T& left, const T& right) {
            return left.first < right.first;
        });
    }

    //return a iter
    Iter find(const Key& t) {
        auto it = std::lower_bound(std::begin(v_), std::end(v_), T(t, Value()), [](const T& left, const T& right) {
            return left.first < right.first;
        });
        if(it == v_.end() || it->first < t) {
            return v_.end();
        }
        else {
            return it;
        }
    }

    std::pair<Iter, Iter> range(const Key& t) {
        auto p = std::equal_range(std::begin(v_), std::end(v_), T(t, Value()), [](const T& left, const T& right) {
            return left.first < right.first;
        });
        return std::move(p);
    }

    int count(const Key& t) {
        auto p = range(t);
        return std::distance(p.first, p.second);
    }

    //insert and erase function, but very very low effective
    void insert(const T& t) {
        auto it = std::lower_bound(std::begin(v_), std::end(v_), t, [](const T& left, const T& right) {
            return left.first < right.first;
        });
        v_.insert(it, t);
    }

    inline void erase(Iter it) {
        v_.erase(it);
    }

    void erase(const Key& t) {
        auto it = find(t);
        erase(it);
    }


private:
    //the meta data
    std::vector<std::pair<Key, Value>> v_;
private:
    inline void init() {
        std::sort(std::begin(v_), std::end(v_), [](const T& left, const T& right) {
            return left.first < right.first;
        });
    }
};

} //wzp


#endif /*BINARY_SEARCH_HPP*/