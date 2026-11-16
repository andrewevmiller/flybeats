// The core, readout and decoder from the reference engine's own ear features: rates bit for bit.
#include "data.h"
#include "flybrain/network.h"
using flybrain::PeakDecoder;

static void compareFeatures(const BrainFile& b, const std::string& dir, const std::string& name, bool dbl, bool f32feats) {
    std::vector<double> ear;
    if (f32feats) { auto f = loadRaw<float>(dir + "/" + name + "_ear.f32"); ear.assign(f.begin(), f.end()); }
    else ear = loadRaw<double>(dir + "/" + name + "_ear.f64");
    std::vector<double> wantRates, wantProb;
    if (f32feats) { auto r = loadRaw<float>(dir + "/" + name + "_rates.f32"); wantRates.assign(r.begin(), r.end()); auto p = loadRaw<float>(dir + "/" + name + "_prob.f32"); wantProb.assign(p.begin(), p.end()); }
    else { wantRates = loadRaw<double>(dir + "/" + name + "_rates.f64"); wantProb = loadRaw<double>(dir + "/" + name + "_prob.f64"); }
    std::vector<uint8_t> wantHits = loadRaw<uint8_t>(dir + "/" + name + "_hits.u8");
    size_t frames = ear.size() / size_t(b.nBands);
    Engine e(b);
    EngineConfig cfg; cfg.hostRate = 16000; cfg.doublePrecision = dbl; cfg.prerollFrames = 0; cfg.flushDenormals = false;
    e.prepare(cfg);
    Snapshot s; e.allocateSnapshot(s);
    std::vector<double> rates, prob;
    std::vector<uint8_t> hits;
    HitEvent ev[64];
    auto take = [&]() { while (e.readSnapshot(s)) { rates.insert(rates.end(), s.rates.begin(), s.rates.end()); prob.insert(prob.end(), s.prob.begin(), s.prob.end()); hits.insert(hits.end(), s.hits.begin(), s.hits.end()); } };
    for (size_t f = 0; f < frames; ++f) { e.stepFeatures(ear.data() + f * size_t(b.nBands), ev, 64); take(); }
    e.flush(ev, 64); take();
    CHECK_MSG(rates.size() == wantRates.size(), "%s: %zu vs %zu rate values", name.c_str(), rates.size(), wantRates.size());
    size_t cnt = std::min(rates.size(), wantRates.size());
    bool bitExact = std::equal(rates.begin(), rates.begin() + long(cnt), wantRates.begin());
    double dr = maxAbsDiff(rates, wantRates, cnt), dp = maxAbsDiff(prob, wantProb, std::min(prob.size(), wantProb.size()));
    bool probExact = std::equal(prob.begin(), prob.begin() + long(std::min(prob.size(), wantProb.size())), wantProb.begin());
    int bad = unexplainedHitDiffs(hits, wantHits, wantProb, b.nDrums, b.hitThreshold);
    std::printf("%s (%s): rates %s (max diff %.3e), prob %s (max diff %.3e), hits %d vs %d, unexplained diffs %d\n", name.c_str(),
                dbl ? "double" : "float", bitExact ? "BIT-EXACT" : "differ", dr, probExact ? "BIT-EXACT" : "differ", dp,
                countHits(hits), countHits(wantHits), bad);
    CHECK_MSG(bitExact, "%s: rates not bit-exact, max diff %.3e", name.c_str(), dr);
    CHECK_MSG(dp <= 1e-5, "%s: prob diff %.3e", name.c_str(), dp);
    CHECK_MSG(bad == 0, "%s: %d unexplained hit differences", name.c_str(), bad);
}

// The streaming decoder against the offline rule of loss.peaks on constructed probabilities: plateaus (ties),
// equality at the threshold, a hit on the first frame, a plateau that runs into the flush. Exact expected hits,
// no ambiguity rule (red team, item 2).
static void decoderTies() {
    const int nd = 3, T_ = 40;
    std::vector<std::vector<double>> p(size_t(T_), std::vector<double>(size_t(nd), 0.1));
    for (int t = 0; t < T_; ++t) {
        p[size_t(t)][0] = (t >= 5 && t < 12) ? 0.9 : (t == 20 ? 0.5 : (t == 21 ? 0.5 : 0.1));   // plateau 5..11; ties at the threshold 20, 21
        p[size_t(t)][1] = (t == 0) ? 0.7 : (t >= 30 ? 0.8 : 0.2);                               // hit at frame 0; plateau to the end
        p[size_t(t)][2] = 0.5;                                                                   // flat at the threshold throughout
    }
    // offline rule: p[t] >= thr, p[t] >= p[t-1] (p[-1] = -inf), p[t] > p[t+1] (p[T] = -inf)
    std::vector<std::vector<int>> want(size_t(T_), std::vector<int>(size_t(nd), 0));
    for (int d = 0; d < nd; ++d)
        for (int t = 0; t < T_; ++t) {
            double left = t > 0 ? p[size_t(t - 1)][size_t(d)] : -1e300, right = t + 1 < T_ ? p[size_t(t + 1)][size_t(d)] : -1e300;
            double x = p[size_t(t)][size_t(d)];
            want[size_t(t)][size_t(d)] = (x >= 0.5 && x >= left && x > right) ? 1 : 0;
        }
    PeakDecoder<double> dec; dec.prepare(nd); dec.reset();
    std::vector<std::vector<int>> got(size_t(T_), std::vector<int>(size_t(nd), -1));
    std::vector<double> vel(size_t(nd), 0.0);
    for (int t = 0; t < T_; ++t)
        if (dec.push(p[size_t(t)].data(), vel.data(), t, 0.5)) {
            CHECK(dec.decidedFrame == t - 1);
            for (int d = 0; d < nd; ++d) got[size_t(dec.decidedFrame)][size_t(d)] = dec.hits[size_t(d)];
        }
    CHECK(dec.flush(0.5) && dec.decidedFrame == T_ - 1);
    for (int d = 0; d < nd; ++d) got[size_t(T_ - 1)][size_t(d)] = dec.hits[size_t(d)];
    int mismatches = 0, expectedHits = 0;
    for (int t = 0; t < T_; ++t)
        for (int d = 0; d < nd; ++d) { expectedHits += want[size_t(t)][size_t(d)]; if (got[size_t(t)][size_t(d)] != want[size_t(t)][size_t(d)]) ++mismatches; }
    CHECK_MSG(mismatches == 0, "decoder ties: %d mismatches against the offline rule", mismatches);
    CHECK(expectedHits == 5);   // plateau end (11), tie pair end (21), frame 0, plateau into the flush (39), flat-at-threshold end (39)
    std::printf("decoder ties: %d expected hits, %d mismatches\n", expectedHits, mismatches);
}

int main(int, char** argv) {
    decoderTies();
    std::string dir = argv[1];
    if (!exists(dir + "/ref64_ear.f64")) { std::printf("SKIP: no test data\n"); return 0; }
    BrainFile b = BrainFile::load(dir + "/small.flybrain");
    compareFeatures(b, dir, "ref64", true, false);
    compareFeatures(b, dir, "ref32", false, true);
    BrainFile hb = BrainFile::load(dir + "/history.flybrain");
    CHECK(hb.nBlocks() == 4 && hb.historyBlocks[0] == 10);
    compareFeatures(hb, dir, "hist64", true, false);
    return finish();
}
