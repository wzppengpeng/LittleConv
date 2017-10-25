#include <cstdio>

#include <string>
#include <uuid/uuid.h>

namespace wzp {

//Generate the uuid
//maybe need to add -luuid
inline static std::string UUID() {
    uuid_t uu;
    uuid_generate( uu );
    std::string res; res.reserve(32);
    char x[10];
    for(int i = 0; i < 16; ++i) {
        sprintf(x, "%02X", uu[i]);
        res.append(std::string(x, x + 2));
    }
    return std::move(res);
}


} //wzp

