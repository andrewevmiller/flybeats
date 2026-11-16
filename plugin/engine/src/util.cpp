#include "flybrain/util.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace flybrain {

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4)
namespace {
const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
    0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
    0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void sha256Block(uint32_t h[8], const uint8_t* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
}  // namespace

std::string sha256Hex(const uint8_t* data, size_t n) {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    size_t i = 0;
    for (; i + 64 <= n; i += 64) sha256Block(h, data + i);
    uint8_t tail[128];
    size_t rem = n - i;
    std::memcpy(tail, data + i, rem);
    tail[rem] = 0x80;
    size_t padLen = (rem < 56) ? 64 : 128;
    std::memset(tail + rem + 1, 0, padLen - rem - 1);
    uint64_t bits = uint64_t(n) * 8;
    for (int k = 0; k < 8; ++k) tail[padLen - 1 - k] = uint8_t(bits >> (8 * k));
    sha256Block(h, tail);
    if (padLen == 128) sha256Block(h, tail + 64);
    char out[65];
    for (int k = 0; k < 8; ++k) std::snprintf(out + 8 * k, 9, "%08x", h[k]);
    return std::string(out, 64);
}

// ---------------------------------------------------------------- JSON
namespace {
struct Parser {
    const std::string& s;
    size_t i = 0;
    explicit Parser(const std::string& text) : s(text) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
    [[noreturn]] void fail(const char* what) { throw std::runtime_error(std::string("json: ") + what + " at " + std::to_string(i)); }
    bool match(const char* lit) {
        size_t n = std::strlen(lit);
        if (s.compare(i, n, lit) == 0) { i += n; return true; }
        return false;
    }
    Json value() {
        ws();
        if (i >= s.size()) fail("unexpected end");
        Json j;
        char c = s[i];
        if (c == '{') {
            j.type = Json::Object; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return j; }
            for (;;) {
                ws();
                if (i >= s.size() || s[i] != '"') fail("expected key");
                std::string k = string();
                ws();
                if (i >= s.size() || s[i] != ':') fail("expected ':'");
                ++i;
                j.obj[k] = value();
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            j.type = Json::Array; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return j; }
            for (;;) {
                j.arr.push_back(value());
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            j.type = Json::String; j.str = string();
        } else if (match("true")) { j.type = Json::Bool; j.b = true; }
        else if (match("false")) { j.type = Json::Bool; j.b = false; }
        else if (match("null")) { j.type = Json::Null; }
        else {
            size_t start = i;
            while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '-' || s[i] == '+' || s[i] == '.' || s[i] == 'e' || s[i] == 'E')) ++i;
            if (start == i) fail("unexpected character");
            j.type = Json::Number;
            j.num = std::strtod(s.substr(start, i - start).c_str(), nullptr);
        }
        return j;
    }
    std::string string() {
        ++i;  // opening quote
        std::string out;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\') {
                ++i;
                if (i >= s.size()) fail("bad escape");
                char e = s[i++];
                switch (e) {
                    case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                    case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break; case 't': out += '\t'; break;
                    case 'u': {
                        if (i + 4 > s.size()) fail("bad \\u");
                        unsigned cp = unsigned(std::strtoul(s.substr(i, 4).c_str(), nullptr, 16));
                        i += 4;
                        if (cp < 0x80) out += char(cp);
                        else if (cp < 0x800) { out += char(0xC0 | (cp >> 6)); out += char(0x80 | (cp & 0x3F)); }
                        else { out += char(0xE0 | (cp >> 12)); out += char(0x80 | ((cp >> 6) & 0x3F)); out += char(0x80 | (cp & 0x3F)); }
                        break;
                    }
                    default: fail("bad escape");
                }
            } else out += s[i++];
        }
        if (i >= s.size()) fail("unterminated string");
        ++i;
        return out;
    }
};
}  // namespace

Json Json::parse(const std::string& text) {
    Parser p(text);
    Json j = p.value();
    p.ws();
    if (p.i != text.size()) p.fail("trailing characters");
    return j;
}

const Json& Json::operator[](const std::string& k) const {
    if (type != Object) throw std::runtime_error("json: not an object, asked for " + k);
    auto it = obj.find(k);
    if (it == obj.end()) throw std::runtime_error("json: missing key " + k);
    return it->second;
}
const Json& Json::at(size_t i) const {
    if (type != Array || i >= arr.size()) throw std::runtime_error("json: bad array index");
    return arr[i];
}
double Json::asNumber() const {
    if (type != Number) throw std::runtime_error("json: not a number");
    return num;
}
int64_t Json::asInt() const { return static_cast<int64_t>(std::llround(asNumber())); }
const std::string& Json::asString() const {
    if (type != String) throw std::runtime_error("json: not a string");
    return str;
}

static void dumpString(std::string& o, const std::string& s) {
    o += '"';
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) { char b[8]; std::snprintf(b, 8, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    o += '"';
}

std::string Json::dump() const {
    std::string o;
    switch (type) {
        case Null: return "null";
        case Bool: return b ? "true" : "false";
        case Number: { char buf[32]; std::snprintf(buf, 32, "%.17g", num); return buf; }
        case String: dumpString(o, str); return o;
        case Array:
            o += '[';
            for (size_t i = 0; i < arr.size(); ++i) { if (i) o += ','; o += arr[i].dump(); }
            return o + ']';
        case Object:
            o += '{';
            { bool first = true; for (auto& kv : obj) { if (!first) o += ','; first = false; dumpString(o, kv.first); o += ':'; o += kv.second.dump(); } }
            return o + '}';
    }
    return o;
}

std::vector<uint8_t> readFile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> out;
    uint8_t buf[1 << 16];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.insert(out.end(), buf, buf + n);
    std::fclose(f);
    return out;
}

double dbToGain(double db) { return std::pow(10.0, db / 20.0); }

}  // namespace flybrain
