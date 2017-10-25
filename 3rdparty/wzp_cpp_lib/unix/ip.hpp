#ifndef IP_WZP_HPP_
#define IP_WZP_HPP_

#include <string>
#include <unordered_set>

#include <net/if.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>

namespace wzp
{

inline static void get_localip_address(std::unordered_set<std::string>& result) {
    result.clear();
    struct ifaddrs* if_addr_struct = nullptr;
    struct ifaddrs* ifa = nullptr;
    void* tmp_addr_ptr = nullptr;

    getifaddrs(&if_addr_struct);
    for (ifa = if_addr_struct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!ifa->ifa_addr) continue;

        if (ifa->ifa_addr->sa_family == AF_INET &&
        (ifa->ifa_flags & IFF_LOOPBACK) == 0) {
            tmp_addr_ptr = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
            char address_buffer[INET_ADDRSTRLEN];
            inet_ntop(AF_INET, tmp_addr_ptr, address_buffer, INET_ADDRSTRLEN);

            std::string ip(address_buffer);
            result.emplace(ip);
        }
    }
    if (if_addr_struct != nullptr) freeifaddrs(if_addr_struct);
}

} //wzp

#endif /*IP_WZP_HPP_*/