// Small self-contained helpers: SHA-256, a minimal JSON reader, byte reading. No dependencies.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace flybrain {

std::string sha256Hex(const uint8_t* data, size_t n);

// JSON value: the subset a brain-file header and a controls file use (objects, arrays, strings, numbers, bools, null).
struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::map<std::string, Json> obj;

    bool has(const std::string& k) const { return type == Object && obj.count(k) != 0; }
    const Json& operator[](const std::string& k) const;
    const Json& at(size_t i) const;
    double asNumber() const;
    int64_t asInt() const;
    const std::string& asString() const;
    size_t size() const { return type == Array ? arr.size() : type == Object ? obj.size() : 0; }
    static Json parse(const std::string& text);
    std::string dump() const;   // compact, for sidecars
};

inline uint32_t readU32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
inline uint64_t readU64(const uint8_t* p) { return uint64_t(readU32(p)) | uint64_t(readU32(p + 4)) << 32; }

std::vector<uint8_t> readFile(const std::string& path);
double dbToGain(double db);

}  // namespace flybrain
