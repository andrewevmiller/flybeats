#include "flybrain/brainfile.h"

#include <cmath>
#include <cstring>

namespace flybrain {

namespace {
struct ArraySpec {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t offset = 0, nbytes = 0;
};

template <class T>
std::vector<T> takeArray(const std::vector<uint8_t>& bytes, const std::map<std::string, ArraySpec>& index,
                         const std::string& name, const char* dtype, int ndim) {
    auto it = index.find(name);
    if (it == index.end()) throw std::runtime_error("brain file: missing array " + name);
    const ArraySpec& a = it->second;
    if (a.dtype != dtype) throw std::runtime_error("brain file: array " + name + " has dtype " + a.dtype + ", want " + dtype);
    if (int(a.shape.size()) != ndim) throw std::runtime_error("brain file: array " + name + " has the wrong number of dimensions");
    uint64_t count = 1;
    for (int64_t s : a.shape) {
        if (s < 0) throw std::runtime_error("brain file: negative shape in " + name);
        count *= uint64_t(s);
    }
    if (count * sizeof(T) != a.nbytes) throw std::runtime_error("brain file: array " + name + " size disagrees with its shape");
    std::vector<T> out(count);
    if (count) std::memcpy(out.data(), bytes.data() + a.offset, a.nbytes);   // little-endian host assumed (checked below)
    return out;
}

bool hostIsLittleEndian() {
    uint16_t x = 1;
    uint8_t b;
    std::memcpy(&b, &x, 1);
    return b == 1;
}
}  // namespace

BrainFile BrainFile::load(const std::string& path) { return parse(readFile(path)); }

BrainFile BrainFile::parse(const std::vector<uint8_t>& bytes) {
    if (!hostIsLittleEndian()) throw std::runtime_error("brain file: big-endian hosts are not supported");
    if (bytes.size() < 20 + 32 || std::memcmp(bytes.data(), "FLYBRAIN", 8) != 0)
        throw std::runtime_error("brain file: not a .flybrain file (bad magic)");
    uint32_t version = readU32(bytes.data() + 8);
    if (version != kFormatVersion)
        throw std::runtime_error("brain file: format version " + std::to_string(version) + ", this reader supports " + std::to_string(kFormatVersion));
    size_t bodyLen = bytes.size() - 32;
    std::string digest = sha256Hex(bytes.data(), bodyLen);
    char trailer[65];
    for (int i = 0; i < 32; ++i) std::snprintf(trailer + 2 * i, 3, "%02x", bytes[bodyLen + i]);
    if (digest != std::string(trailer, 64)) throw std::runtime_error("brain file: hash mismatch, the file is corrupt or was edited");
    uint64_t headerLen = readU64(bytes.data() + 12);
    if (20 + headerLen > bodyLen) throw std::runtime_error("brain file: header runs past the end of the file");

    BrainFile b;
    b.sha256 = digest;
    b.headerJson.assign(reinterpret_cast<const char*>(bytes.data() + 20), size_t(headerLen));
    Json header = Json::parse(b.headerJson);
    b.meta = header["meta"];
    std::map<std::string, ArraySpec> index;
    for (auto& kv : header["arrays"].obj) {
        ArraySpec a;
        a.dtype = kv.second["dtype"].asString();
        for (auto& s : kv.second["shape"].arr) a.shape.push_back(s.asInt());
        a.offset = uint64_t(kv.second["offset"].asInt());
        a.nbytes = uint64_t(kv.second["nbytes"].asInt());
        if (a.offset < 20 + headerLen || a.offset + a.nbytes > bodyLen)
            throw std::runtime_error("brain file: array " + kv.first + " lies outside the file");
        index[kv.first] = a;
    }
    const Json& m = b.meta;
    b.name = m["name"].asString();
    b.capability = m["capability"].asString();
    b.inputTask = m["input_task"].asString();
    if (b.capability != "observation_only" && b.capability != "drum_capable")
        throw std::runtime_error("brain file: unknown capability " + b.capability);
    b.n = int(m["n_neurons"].asInt());
    b.nnz = int(m["n_edges"].asInt());
    b.nDrums = int(m["n_drums"].asInt());
    b.nBands = int(m["n_bands"].asInt());
    b.nFft = int(m["n_fft"].asInt());
    b.hop = int(m["hop"].asInt());
    b.frameMs = m["frame_ms"].asNumber();
    b.hitThreshold = m["hit_threshold"].asNumber();
    b.tauMin = m["tau_min_ms"].asNumber();
    b.tauMax = m["tau_max_ms"].asNumber();
    for (auto& p : m["pieces"].arr) b.pieces.push_back(p.asString());
    if (int(b.pieces.size()) != b.nDrums) throw std::runtime_error("brain file: pieces disagree with n_drums");
    for (auto& p : b.pieces) {
        std::vector<int> notes;
        if (m["gm_notes"].has(p)) for (auto& v : m["gm_notes"][p].arr) notes.push_back(int(v.asInt()));
        b.gmNotes.push_back(notes);
    }
    if (m.has("runaway") && m["runaway"].has("mean_rate_limit")) {
        b.runaway.set = true;
        b.runaway.meanRateLimit = m["runaway"]["mean_rate_limit"].asNumber();
        b.runaway.rateCeiling = m["runaway"]["rate_ceiling"].asNumber();
        b.runaway.frames = int(m["runaway"]["frames"].asInt());
    }

    b.wIndptr = takeArray<int32_t>(bytes, index, "w_indptr", "int32", 1);
    b.wIndices = takeArray<int32_t>(bytes, index, "w_indices", "int32", 1);
    b.wValues = takeArray<float>(bytes, index, "w_values", "float32", 1);
    b.wUnsigned = takeArray<float>(bytes, index, "w_unsigned", "float32", 1);
    b.wPair = takeArray<int32_t>(bytes, index, "w_pair", "int32", 1);
    b.alpha = takeArray<float>(bytes, index, "alpha", "float32", 1);
    b.tauMs = takeArray<float>(bytes, index, "tau_ms", "float32", 1);
    b.bias = takeArray<float>(bytes, index, "bias", "float32", 1);
    b.alive = takeArray<float>(bytes, index, "alive", "float32", 1);
    b.sign = takeArray<int8_t>(bytes, index, "sign", "int8", 1);
    b.typeIdx = takeArray<int32_t>(bytes, index, "type_idx", "int32", 1);
    b.earIdx = takeArray<int32_t>(bytes, index, "ear_idx", "int32", 1);
    b.earBand = takeArray<int32_t>(bytes, index, "ear_band", "int32", 1);
    b.bandMean = takeArray<float>(bytes, index, "band_mean", "float32", 1);
    b.bandStd = takeArray<float>(bytes, index, "band_std", "float32", 1);
    b.bandGain = takeArray<float>(bytes, index, "band_gain", "float32", 1);
    b.bandOffset = takeArray<float>(bytes, index, "band_offset", "float32", 1);
    b.bandMatrix = takeArray<float>(bytes, index, "band_matrix", "float32", 2);
    b.mnIdx = takeArray<int32_t>(bytes, index, "mn_idx", "int32", 1);
    b.mnDrum = takeArray<int32_t>(bytes, index, "mn_drum", "int32", 1);
    b.mnMean = takeArray<float>(bytes, index, "mn_mean", "float32", 1);
    b.mnStd = takeArray<float>(bytes, index, "mn_std", "float32", 1);
    b.wHit = takeArray<float>(bytes, index, "w_hit", "float32", 1);
    b.uVel = takeArray<float>(bytes, index, "u_vel", "float32", 1);
    b.cHit = takeArray<float>(bytes, index, "c_hit", "float32", 1);
    b.eVel = takeArray<float>(bytes, index, "e_vel", "float32", 1);
    b.historyBlocks = takeArray<int32_t>(bytes, index, "history_blocks", "int32", 1);
    b.pHit = takeArray<float>(bytes, index, "p_hit", "float32", 2);
    b.pVel = takeArray<float>(bytes, index, "p_vel", "float32", 2);
    b.bodyId = takeArray<int64_t>(bytes, index, "body_id", "int64", 1);
    b.hop_ = takeArray<int16_t>(bytes, index, "hop", "int16", 1);
    b.side = takeArray<int8_t>(bytes, index, "side", "int8", 1);
    b.motorGroup = takeArray<int8_t>(bytes, index, "motor_group", "int8", 1);
    b.drumOfNeuron = takeArray<int8_t>(bytes, index, "drum_of_neuron", "int8", 1);

    // consistency (the same checks as verify_brain)
    const size_t n = size_t(b.n), nnz = size_t(b.nnz);
    auto want = [&](size_t got, size_t w, const char* name) {
        if (got != w) throw std::runtime_error(std::string("brain file: ") + name + " has the wrong length");
    };
    for (auto* v : {&b.alpha, &b.tauMs, &b.bias, &b.alive}) want(v->size(), n, "a per-neuron array");
    want(b.sign.size(), n, "sign"); want(b.typeIdx.size(), n, "type_idx"); want(b.bodyId.size(), n, "body_id");
    want(b.hop_.size(), n, "hop"); want(b.side.size(), n, "side"); want(b.motorGroup.size(), n, "motor_group");
    want(b.drumOfNeuron.size(), n, "drum_of_neuron");
    want(b.wIndptr.size(), n + 1, "w_indptr");
    if (b.wIndptr[0] != 0 || size_t(b.wIndptr[n]) != nnz) throw std::runtime_error("brain file: w_indptr does not describe n rows with n_edges entries");
    want(b.wIndices.size(), nnz, "w_indices"); want(b.wValues.size(), nnz, "w_values");
    want(b.wUnsigned.size(), nnz, "w_unsigned"); want(b.wPair.size(), nnz, "w_pair");
    for (size_t r = 0; r < n; ++r) {
        if (b.wIndptr[r + 1] < b.wIndptr[r]) throw std::runtime_error("brain file: w_indptr is not non-decreasing");
        for (int32_t k = b.wIndptr[r]; k < b.wIndptr[r + 1]; ++k) {
            if (b.wIndices[k] < 0 || size_t(b.wIndices[k]) >= n) throw std::runtime_error("brain file: w_indices out of range");
            if (k > b.wIndptr[r] && b.wIndices[k] <= b.wIndices[k - 1]) throw std::runtime_error("brain file: columns not strictly increasing in a row");
            if (!std::isfinite(b.wValues[k])) throw std::runtime_error("brain file: non-finite weight");
        }
    }
    want(b.earBand.size(), b.earIdx.size(), "ear_band");
    for (size_t i = 0; i < b.earIdx.size(); ++i) {
        if (b.earIdx[i] < 0 || size_t(b.earIdx[i]) >= n) throw std::runtime_error("brain file: ear_idx out of range");
        if (b.earBand[i] < 0 || b.earBand[i] >= b.nBands) throw std::runtime_error("brain file: ear_band out of range");
    }
    for (auto* v : {&b.bandMean, &b.bandStd, &b.bandGain, &b.bandOffset}) want(v->size(), size_t(b.nBands), "a band array");
    want(b.bandMatrix.size(), size_t(b.nFft / 2 + 1) * size_t(b.nBands), "band_matrix");
    const size_t nm = b.mnIdx.size();
    for (auto* v : {&b.mnMean, &b.mnStd, &b.wHit, &b.uVel}) want(v->size(), nm, "a readout array");
    want(b.mnDrum.size(), nm, "mn_drum");
    for (size_t i = 0; i < nm; ++i) {
        if (b.mnIdx[i] < 0 || size_t(b.mnIdx[i]) >= n) throw std::runtime_error("brain file: mn_idx out of range");
        if (b.mnDrum[i] < 0 || b.mnDrum[i] >= b.nDrums) throw std::runtime_error("brain file: mn_drum out of range");
    }
    want(b.cHit.size(), size_t(b.nDrums), "c_hit"); want(b.eVel.size(), size_t(b.nDrums), "e_vel");
    const size_t nb = b.historyBlocks.size();
    want(b.pHit.size(), size_t(b.nDrums) * (1 + nb), "p_hit"); want(b.pVel.size(), size_t(b.nDrums) * (1 + nb), "p_vel");
    for (int32_t h : b.historyBlocks) if (h < 1) throw std::runtime_error("brain file: history block under one frame");
    for (size_t i = 0; i < n; ++i) {
        if (!(b.alpha[i] > 0 && b.alpha[i] <= 1)) throw std::runtime_error("brain file: alpha outside (0, 1]");
        if (b.alive[i] == 0 && b.sign[i] != 0) throw std::runtime_error("brain file: a silenced neuron has a non-zero sign");
    }
    if (b.hop <= 0 || b.nFft <= 0 || (b.nFft & (b.nFft - 1)) != 0) throw std::runtime_error("brain file: n_fft must be a power of two");
    return b;
}

}  // namespace flybrain
