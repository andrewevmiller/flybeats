// Shared helpers for the engine tests: raw-array loaders and a frame-by-frame engine run.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "flybrain/engine.h"
#include "flybrain/util.h"

using namespace flybrain;

inline bool exists(const std::string& p) { FILE* f = std::fopen(p.c_str(), "rb"); if (f) std::fclose(f); return f != nullptr; }
template <class T> std::vector<T> loadRaw(const std::string& p) {
    std::vector<uint8_t> raw = readFile(p);
    std::vector<T> out(raw.size() / sizeof(T));
    std::memcpy(out.data(), raw.data(), out.size() * sizeof(T));
    return out;
}

struct Run {
    std::vector<int64_t> frames;
    std::vector<double> rates, prob, vel, ear, bands;   // frames x width
    std::vector<uint8_t> hits, resetFlags, runawayFlags, muted;
    std::vector<HitEvent> events;
    int n = 0, nd = 0, nb = 0;
    int runaways = 0;
};

struct RunOptions {
    int rate = 16000, block = 512, preroll = 0;
    bool dbl = true, ftz = true;
    const Controls* controls = nullptr;
    int controlsAtFrame = -1;   // apply the controls after this many frames (-1: before the start)
    double trimDb = 0.0;
};

// Feeds the engine in slices no longer than one frame and reads the snapshot after each, so every frame is seen.
inline Run runEngine(const BrainFile& b, const std::vector<float>& audio, const RunOptions& o) {
    Engine e(b);
    EngineConfig cfg; cfg.hostRate = o.rate; cfg.maxBlock = std::max(o.block, 1); cfg.doublePrecision = o.dbl;
    cfg.flushDenormals = o.ftz; cfg.prerollFrames = o.preroll; cfg.inputTrimDb = o.trimDb;
    e.prepare(cfg);
    if (o.controls && o.controlsAtFrame < 0) e.setControls(*o.controls);
    Run r; r.n = b.n; r.nd = b.nDrums; r.nb = b.nBands;
    Snapshot s; e.allocateSnapshot(s);
    std::vector<HitEvent> ev(1024);
    auto take = [&]() {
        while (e.readSnapshot(s)) {
            r.frames.push_back(s.frame);
            r.rates.insert(r.rates.end(), s.rates.begin(), s.rates.end());
            r.prob.insert(r.prob.end(), s.prob.begin(), s.prob.end());
            r.vel.insert(r.vel.end(), s.vel.begin(), s.vel.end());
            r.ear.insert(r.ear.end(), s.ear.begin(), s.ear.end());
            r.bands.insert(r.bands.end(), s.bands.begin(), s.bands.end());
            r.hits.insert(r.hits.end(), s.hits.begin(), s.hits.end());
            r.resetFlags.push_back(s.resetFlag); r.runawayFlags.push_back(s.runaway); r.muted.push_back(s.muted);
        }
    };
    int slice = std::max(1, std::min(o.block, int(80LL * o.rate / 16000)));
    bool applied = o.controlsAtFrame < 0;
    for (size_t pos = 0; pos < audio.size(); pos += size_t(slice)) {
        int n = int(std::min(size_t(slice), audio.size() - pos));
        const float* ch[1] = {audio.data() + pos};
        int m = e.process(ch, 1, n, ev.data(), int(ev.size()));
        for (int k = 0; k < m; ++k) { ev[k].sampleOffset += int(pos); r.events.push_back(ev[k]); }
        take();
        if (!applied && o.controls && e.framesProcessed() >= o.controlsAtFrame) { e.setControls(*o.controls); applied = true; }
    }
    int m = e.flush(ev.data(), int(ev.size()));
    for (int k = 0; k < m; ++k) r.events.push_back(ev[k]);
    take();
    r.runaways = e.runawayCount();
    return r;
}

inline double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b, size_t count) {
    double m = 0;
    for (size_t i = 0; i < count; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}
inline double maxAbs(const std::vector<double>& a) { double m = 0; for (double x : a) m = std::max(m, std::fabs(x)); return m; }
inline bool allFinite(const std::vector<double>& a) { for (double x : a) if (!std::isfinite(x)) return false; return true; }

// Hit differences not explained by a near-tie in the reference probabilities (parity.ambiguous_hits).
inline int unexplainedHitDiffs(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, const std::vector<double>& refProb,
                               int nd, double thr = 0.5, double eps = 1e-5) {
    size_t frames = want.size() / size_t(nd);
    int bad = 0;
    for (size_t f = 0; f < frames && f * nd < got.size(); ++f)
        for (int d = 0; d < nd; ++d) {
            if (got[f * nd + d] == want[f * nd + d]) continue;
            double p = refProb[f * nd + d];
            bool nearThr = std::fabs(p - thr) < eps;
            bool nearL = f > 0 && std::fabs(p - refProb[(f - 1) * nd + d]) < eps;
            bool nearR = f + 1 < frames && std::fabs(p - refProb[(f + 1) * nd + d]) < eps;
            if (!((nearThr || nearL || nearR) && p >= thr - eps)) ++bad;
        }
    return bad;
}
inline int countHits(const std::vector<uint8_t>& h) { int c = 0; for (auto x : h) c += x; return c; }
