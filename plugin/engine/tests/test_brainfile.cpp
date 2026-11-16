#include <cstring>
#include <functional>

#include "data.h"
#include "flybrain/brainfile.h"

int main(int, char** argv) {
    std::string dir = argv[1];
    if (!exists(dir + "/small.flybrain")) { std::printf("SKIP: no test data (run tests/data/make_data.py)\n"); return 0; }
    BrainFile b = BrainFile::load(dir + "/small.flybrain");
    CHECK(b.n == 183);
    CHECK(b.nDrums == 8 && b.pieces.size() == 8 && b.pieces[0] == "kick");
    CHECK(b.hop == 80 && b.nFft == 512 && b.nBands == 16);
    CHECK(b.gmNotes[0].size() == 2 && b.gmNotes[0][0] == 35);
    CHECK(b.wIndptr.size() == size_t(b.n + 1) && int(b.wValues.size()) == b.nnz);
    CHECK(b.capability == "observation_only" && b.inputTask == "unknown");
    CHECK(b.sha256.size() == 64);
    CHECK(b.headerJson.find("\"meta\"") != std::string::npos);
    // signed == sign x unsigned
    bool ok = true;
    for (int k = 0; k < b.nnz; ++k) ok = ok && b.wValues[k] == float(b.sign[b.wIndices[k]]) * b.wUnsigned[k];
    CHECK(ok);

    std::vector<uint8_t> bytes = readFile(dir + "/small.flybrain");
    auto corrupt = bytes; corrupt[corrupt.size() / 2] ^= 0xFF;
    CHECK(throwsWith([&] { BrainFile::parse(corrupt); }, "hash"));
    auto magic = bytes; std::memcpy(magic.data(), "RIFF", 4);
    CHECK(throwsWith([&] { BrainFile::parse(magic); }, "magic"));
    auto version = bytes; version[8] = 2;   // the hash then fails too, but the version is checked first
    CHECK(throwsWith([&] { BrainFile::parse(version); }, "version"));
    auto trunc = bytes; trunc.resize(100);
    CHECK(throwsWith([&] { BrainFile::parse(trunc); }, "brain file"));
    CHECK(throwsWith([&] { BrainFile::load(dir + "/bad_index.flybrain"); }, "w_indices"));
    CHECK(throwsWith([&] { BrainFile::load(dir + "/bad_alpha.flybrain"); }, "alpha"));
    CHECK(throwsWith([&] { BrainFile::load(dir + "/does_not_exist.flybrain"); }, "cannot open"));
    // SHA-256 known answers
    CHECK(sha256Hex(reinterpret_cast<const uint8_t*>("abc"), 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(sha256Hex(nullptr, 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    std::string s56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(sha256Hex(reinterpret_cast<const uint8_t*>(s56.data()), s56.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // JSON
    Json j = Json::parse(R"({"a":[1,2.5,-3e2],"b":"x\"yé","c":true,"d":null,"e":{}})");
    CHECK(j["a"].size() == 3 && j["a"].at(2).asNumber() == -300.0);
    CHECK(j["b"].asString() == "x\"y\xc3\xa9");
    CHECK(j["c"].b && j["d"].type == Json::Null && j["e"].size() == 0);
    CHECK(throwsWith([&] { Json::parse("{\"a\":}"); }, "json"));
    return finish();
}
