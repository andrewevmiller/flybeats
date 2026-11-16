// Reader for the .flybrain v1 container (plugin/export/flybrain/brainfile.py is the writer and the spec).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "flybrain/util.h"

namespace flybrain {

constexpr uint32_t kFormatVersion = 1;

struct RunawayLimits {
    bool set = false;
    double rateCeiling = 0, meanRateLimit = 0;
    int frames = 20;
};

struct BrainFile {
    // identity and header
    std::string sha256;            // hex of everything before the trailer: the brain's identity
    std::string headerJson;        // the whole header text, for the UI
    Json meta;                     // header["meta"]
    std::string name, capability, inputTask;

    // sizes and constants
    int n = 0, nnz = 0, nDrums = 0, nBands = 0, nFft = 0, hop = 0;
    double frameMs = 5, hitThreshold = 0.5, tauMin = 10, tauMax = 200;
    std::vector<std::string> pieces;
    std::vector<std::vector<int>> gmNotes;   // per drum
    RunawayLimits runaway;

    // core (CSR by receiving neuron)
    std::vector<int32_t> wIndptr, wIndices, wPair;
    std::vector<float> wValues, wUnsigned, alpha, tauMs, bias, alive;
    std::vector<int8_t> sign;
    std::vector<int32_t> typeIdx;
    // ear
    std::vector<int32_t> earIdx, earBand;
    std::vector<float> bandMean, bandStd, bandGain, bandOffset, bandMatrix;  // bandMatrix (nFft/2+1) x nBands row-major
    // readout
    std::vector<int32_t> mnIdx, mnDrum, historyBlocks;
    std::vector<float> mnMean, mnStd, wHit, uVel, cHit, eVel, pHit, pVel;    // pHit/pVel nDrums x (1 + blocks)
    // display
    std::vector<int64_t> bodyId;
    std::vector<int16_t> hop_;
    std::vector<int8_t> side, motorGroup, drumOfNeuron;

    int nEars() const { return int(earIdx.size()); }
    int nMn() const { return int(mnIdx.size()); }
    int nBlocks() const { return int(historyBlocks.size()); }

    // Throws std::runtime_error with the reason on any refusal.
    static BrainFile load(const std::string& path);
    static BrainFile parse(const std::vector<uint8_t>& bytes);
};

}  // namespace flybrain
