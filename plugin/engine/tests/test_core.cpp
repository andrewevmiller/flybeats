// The core, readout and decoder from the reference engine's own ear features: rates bit for bit.
#include "data.h"

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

int main(int, char** argv) {
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
