#pragma once
#include <exception>

namespace neuronet {
    namespace exception {
        struct MissingInputLayer : public std::exception {
            public:
                const char* what() {
                    return static_cast<const char *>("Input layer is missing");
                }
        };
        struct MissingHiddenLayer : public std::exception {
            public:
                const char* what() {
                    return static_cast<const char *>("Hidden layer/s is missing");
                }
        };
        struct MissingOuputLayer : public std::exception {
            public:
                const char* what() {
                    return static_cast<const char *>("Output layer is missing");
                }
        };
    }
}