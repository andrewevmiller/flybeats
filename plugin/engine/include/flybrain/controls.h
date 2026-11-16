// The control block the UI fills (section 6 of the plan). Plain data; the engine compiles it per precision.
#pragma once
#include <cstdint>
#include <vector>

#include "flybrain/brainfile.h"

namespace flybrain {

constexpr size_t kMaxInjections = 64;

struct Injection {
    int neuron = 0;
    float current = 0;
    int64_t startFrame = -1;   // -1: from the next frame after the swap
    int frames = 0;
};

struct Controls {
    std::vector<float> lesion;       // n, 1 = alive
    std::vector<float> earMute;      // nEars, 1 = hears
    double gainScale = 1.0;
    double tauScale = 1.0;
    std::vector<float> signFlip;     // n, +1 / -1
    std::vector<int8_t> unsilence;   // n, 0 or +1 / -1 for silenced senders
    std::vector<float> bandOffset;   // nBands
    Injection inject[kMaxInjections];
    size_t nInject = 0;
    double threshold = 0.5;

    explicit Controls(const BrainFile& b) { reset(b); }
    void reset(const BrainFile& b) {
        lesion.assign(size_t(b.n), 1.0f);
        earMute.assign(size_t(b.nEars()), 1.0f);
        gainScale = tauScale = 1.0;
        signFlip.assign(size_t(b.n), 1.0f);
        unsilence.assign(size_t(b.n), 0);
        bandOffset.assign(size_t(b.nBands), 0.0f);
        nInject = 0;
        threshold = b.hitThreshold;
    }
    bool addInjection(const Injection& inj) {
        if (nInject >= kMaxInjections) return false;
        inject[nInject++] = inj;
        return true;
    }
    bool isNeutral(const BrainFile& b) const {
        for (float x : lesion) if (x != 1.0f) return false;
        for (float x : earMute) if (x != 1.0f) return false;
        for (float x : signFlip) if (x != 1.0f) return false;
        for (int8_t x : unsilence) if (x != 0) return false;
        for (float x : bandOffset) if (x != 0.0f) return false;
        return gainScale == 1.0 && tauScale == 1.0 && nInject == 0 && threshold == b.hitThreshold;
    }
};

}  // namespace flybrain
