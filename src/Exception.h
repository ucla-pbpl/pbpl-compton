// -*- mode: c++; c-file-style: "stroustrup"; c-basic-offset: 4 -*-
#ifndef PBPL_Exception_h
#define PBPL_Exception_h

#include <exception>
#include <string>

namespace PBPL {

class Exception : public std::runtime_error {
    std::string msg;
public:
    Exception(const std::string &arg, const char *file, int line) :
        std::runtime_error(arg) {
        std::ostringstream o;
        o << file << ":" << line << ": " << arg;
        msg = o.str();
    }
    ~Exception() throw() {}
    const char *what() const throw() {
        return msg.c_str();
    }
};

}

#define pbpl_throw(arg) throw PBPL::Exception(arg, __FILE__, __LINE__);

#endif
