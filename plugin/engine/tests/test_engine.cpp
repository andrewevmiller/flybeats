// The whole engine against the reference engine, block sizes, host rates, stability, controls, the guard, pre-roll.
#include <random>

#include "data.h"

static const double RATE_ABS = 1e-6, RATE_REL = 1e-5, PROB_ABS = 1e-5;

struct Ref { std::vector<double> rates, prob, ear; std::vector<uint8_t> hits; };
static Ref loadRef(const std::string& dir, const std::string& name) {
    Ref r;
    r.rates = loadRaw<double>(dir + "/" + name + "_rates.f64");
    r.prob = loadRaw<double>(dir + "/" + name + "_prob.f64");
    r.hits = loadRaw<uint8_t>(dir + "/" + name + "_hits.u8");
    if (exists(dir + "/" + name + "_ear.f64")) r.ear = loadRaw<double>(dir + "/" + name + "_ear.f64");
    return r;
}

static bool withinTolerance(const Run& got, const Ref& want, int nd, double thr, const char* label) {
    size_t cnt = std::min(got.rates.size(), want.rates.size());
    double dr = maxAbsDiff(got.rates, want.rates, cnt), limit = RATE_ABS + RATE_REL * maxAbs(want.rates);
    double dp = maxAbsDiff(got.prob, want.prob, std::min(got.prob.size(), want.prob.size()));
    int bad = unexplainedHitDiffs(got.hits, want.hits, want.prob, nd, thr);
    std::printf("%s: frames %zu/%zu, rate err %.3e (limit %.3e), prob err %.3e, hits %d vs %d, unexplained %d\n", label,
                got.frames.size(), want.hits.size() / size_t(nd), dr, limit, dp, countHits(got.hits), countHits(want.hits), bad);
    bool ok = got.rates.size() == want.rates.size() && dr <= limit && dp <= PROB_ABS && bad == 0;
    CHECK_MSG(ok, "%s", label);
    return ok;
}

int main(int, char** argv) {
    std::string dir = argv[1];
    if (!exists(dir + "/audio16k.f32")) { std::printf("SKIP: no test data\n"); return 0; }
    BrainFile b = BrainFile::load(dir + "/small.flybrain");
    std::vector<float> audio = loadRaw<float>(dir + "/audio16k.f32");
    const double thr = b.hitThreshold;

    // 1. full chain at 16 kHz against the reference (double and float), with and without FTZ
    Ref ref64 = loadRef(dir, "ref64");
    RunOptions o; o.rate = 16000; o.block = 512; o.dbl = true;
    Run base = runEngine(b, audio, o);
    withinTolerance(base, ref64, b.nDrums, thr, "16k double vs reference f64");
    o.ftz = false; withinTolerance(runEngine(b, audio, o), ref64, b.nDrums, thr, "16k double no-FTZ vs reference f64"); o.ftz = true;
    o.dbl = false; Run basef = runEngine(b, audio, o); withinTolerance(basef, ref64, b.nDrums, thr, "16k float vs reference f64"); o.dbl = true;
    CHECK(base.frames.size() == audio.size() / 80);
    for (size_t i = 0; i < base.frames.size(); ++i) CHECK(base.frames[i] == int64_t(i));
    CHECK(base.resetFlags[0] == 1 && base.resetFlags[1] == 0);
    // hit events agree with the snapshot's hits
    { int fromEvents = int(base.events.size()); CHECK_MSG(fromEvents == countHits(base.hits), "%d events, %d hits", fromEvents, countHits(base.hits)); }
    // the ear path equals the reference's ear array within tolerance
    { double de = maxAbsDiff(base.ear, ref64.ear, std::min(base.ear.size(), ref64.ear.size())); CHECK_MSG(de < 1e-9, "ear %.3e", de); }

    // 2. block sizes: bit-identical
    for (int block : {1, 64, 480, 4096}) {
        RunOptions ob = o; ob.block = block;
        Run r = runEngine(b, audio, ob);
        CHECK_MSG(r.rates == base.rates && r.prob == base.prob && r.hits == base.hits, "block %d differs", block);
    }
    // 2b. a whole-block run (no slicing) gives the same events and the same last snapshot
    {
        Engine e(b); EngineConfig cfg; cfg.hostRate = 16000; cfg.maxBlock = 4096; cfg.doublePrecision = true; cfg.prerollFrames = 0; e.prepare(cfg);
        std::vector<HitEvent> ev(4096); std::vector<HitEvent> all;
        for (size_t p = 0; p < audio.size(); p += 4096) {
            const float* ch[1] = {audio.data() + p};
            int m = e.process(ch, 1, int(std::min(size_t(4096), audio.size() - p)), ev.data(), int(ev.size()));
            for (int k = 0; k < m; ++k) { ev[k].sampleOffset += int(p); all.push_back(ev[k]); }
        }
        int m = e.flush(ev.data(), int(ev.size()));
        for (int k = 0; k < m; ++k) all.push_back(ev[k]);
        bool same = all.size() == base.events.size();
        for (size_t k = 0; same && k < all.size(); ++k) same = all[k].frame == base.events[k].frame && all[k].drum == base.events[k].drum && all[k].velocity == base.events[k].velocity && all[k].sampleOffset == base.events[k].sampleOffset;
        CHECK_MSG(same, "whole-block events differ (%zu vs %zu)", all.size(), base.events.size());
        // a hit for frame f becomes known at the last sample of frame f+1: offset == (f+2)*80 - 1
        bool offsets = true;
        for (auto& h : all) if (h.sampleOffset >= 0) offsets = offsets && h.sampleOffset == int((h.frame + 2) * 80 - 1);
        CHECK(offsets);
        CHECK(e.latencyHostSamples() == 160);
    }

    // 3. host rates: the reference ran on the numpy port's 16 kHz audio
    for (int rate : {44100, 48000, 96000}) {
        std::string name = "host" + std::to_string(rate);
        if (!exists(dir + "/" + name + ".f32")) continue;
        std::vector<float> x = loadRaw<float>(dir + "/" + name + ".f32");
        RunOptions oh = o; oh.rate = rate; oh.block = 512;
        Run r = runEngine(b, x, oh);
        Ref want = loadRef(dir, name);
        // the resampler's own difference (1e-6 in the samples) enters the rates; measure and require the tolerance
        withinTolerance(r, want, b.nDrums, thr, (name + " double vs reference on numpy-resampled audio").c_str());
        Engine e(b); EngineConfig cfg; cfg.hostRate = rate; e.prepare(cfg);
        std::printf("  latency at %d Hz: %d host samples = %.2f ms\n", rate, e.latencyHostSamples(), 1000.0 * e.latencyHostSamples() / rate);
    }
    // stereo input is averaged: (x, x) equals x; (x, -x) is silence
    {
        std::vector<float> neg(audio.size()); for (size_t i = 0; i < neg.size(); ++i) neg[i] = -audio[i];
        Engine e(b); EngineConfig cfg; cfg.hostRate = 16000; cfg.doublePrecision = true; cfg.prerollFrames = 0; e.prepare(cfg);
        Snapshot s; e.allocateSnapshot(s); HitEvent ev[64];
        const float* ch[2] = {audio.data(), audio.data()};
        std::vector<double> rates;
        for (size_t p = 0; p < audio.size(); p += 80) { const float* c2[2] = {ch[0] + p, ch[1] + p}; e.process(c2, 2, 80, ev, 64); while (e.readSnapshot(s)) rates.insert(rates.end(), s.rates.begin(), s.rates.end()); }
        e.flush(ev, 64); while (e.readSnapshot(s)) rates.insert(rates.end(), s.rates.begin(), s.rates.end());
        CHECK(rates == base.rates);
        e.reset();
        std::vector<double> silent, first;
        for (size_t p = 0; p < audio.size(); p += 80) { const float* c2[2] = {audio.data() + p, neg.data() + p}; e.process(c2, 2, 80, ev, 64); while (e.readSnapshot(s)) { silent.insert(silent.end(), s.ear.begin(), s.ear.end()); if (first.empty()) first = s.ear; } }
        bool allSame = true; for (size_t i = 0; i < silent.size(); ++i) allSame = allSame && silent[i] == first[i % 16];
        CHECK(allSame);
    }

    // 4. stability: 60 s of silence, white noise and a clipped square, no NaN/Inf, float and double
    {
        std::mt19937 rng(2); std::normal_distribution<float> nd(0.0f, 0.3f);
        std::vector<float> silence(16000 * 60, 0.0f), noise(16000 * 60), square(16000 * 60);
        for (auto& v : noise) v = nd(rng);
        for (size_t i = 0; i < square.size(); ++i) square[i] = (i / 40) % 2 ? 1.0f : -1.0f;
        for (bool dbl : {true, false}) {
            for (auto* sig : {&silence, &noise, &square}) {
                Engine e(b); EngineConfig cfg; cfg.hostRate = 16000; cfg.maxBlock = 512; cfg.doublePrecision = dbl; cfg.prerollFrames = 0; e.prepare(cfg);
                Snapshot s; e.allocateSnapshot(s); HitEvent ev[256];
                bool finite = true; double peak = 0; long hits = 0;
                for (size_t p = 0; p < sig->size(); p += 512) { const float* ch[1] = {sig->data() + p}; hits += e.process(ch, 1, 512, ev, 256); if (e.readSnapshot(s)) { finite = finite && allFinite(s.rates) && allFinite(s.prob); peak = std::max(peak, maxAbs(s.rates)); } }
                CHECK_MSG(finite, "%s %s not finite", dbl ? "double" : "float", sig == &silence ? "silence" : sig == &noise ? "noise" : "square");
                std::printf("stability %s %s: 60 s, finite, peak rate %.4g, %ld hits, runaways %d\n", dbl ? "double" : "float", sig == &silence ? "silence" : sig == &noise ? "noise" : "square", peak, hits, e.runawayCount());
            }
        }
    }

    // 5. neutral controls equal the baseline bit for bit; each perturbation equals the reference
    {
        Controls neutral(b);
        RunOptions on = o; on.controls = &neutral;
        Run r = runEngine(b, audio, on);
        CHECK(r.rates == base.rates && r.hits == base.hits);
        const char* kinds[] = {"lesion", "deafen", "gain", "tau", "sign_flip", "band_offset", "unsilence", "inject"};
        for (const char* kind : kinds) {
            std::string jp = dir + "/pert_" + kind + ".json";
            if (!exists(jp)) continue;
            auto raw = readFile(jp);
            Json j = Json::parse(std::string(raw.begin(), raw.end()));
            Controls c(b);
            if (j.has("lesion")) for (auto& v : j["lesion"].arr) c.lesion[size_t(v.asInt())] = 0.0f;
            if (j.has("ear_mute")) for (auto& v : j["ear_mute"].arr) c.earMute[size_t(v.asInt())] = 0.0f;
            if (j.has("gain_scale")) c.gainScale = j["gain_scale"].asNumber();
            if (j.has("tau_scale")) c.tauScale = j["tau_scale"].asNumber();
            if (j.has("sign_flip")) for (auto& v : j["sign_flip"].arr) c.signFlip[size_t(v.asInt())] = -1.0f;
            if (j.has("unsilence")) for (auto& v : j["unsilence"].arr) c.unsilence[size_t(v.at(0).asInt())] = int8_t(v.at(1).asInt());
            if (j.has("band_offset")) for (size_t k = 0; k < 16; ++k) c.bandOffset[k] = float(j["band_offset"].at(k).asNumber());
            if (j.has("inject")) for (auto& v : j["inject"].arr) { Injection inj; inj.neuron = int(v.at(0).asInt()); inj.current = float(v.at(1).asNumber()); inj.startFrame = v.at(2).asInt(); inj.frames = int(v.at(3).asInt()); c.addInjection(inj); }
            CHECK(!c.isNeutral(b));
            RunOptions op = o; op.controls = &c;
            Run rp = runEngine(b, audio, op);
            Ref want; want.rates = loadRaw<double>(dir + "/pert_" + kind + "_rates.f64"); want.hits = loadRaw<uint8_t>(dir + "/pert_" + kind + "_hits.u8");
            size_t cnt = std::min(rp.rates.size(), want.rates.size());
            bool exact = rp.rates.size() == want.rates.size() && std::equal(rp.rates.begin(), rp.rates.end(), want.rates.begin());
            double dr = maxAbsDiff(rp.rates, want.rates, cnt), limit = RATE_ABS + RATE_REL * maxAbs(want.rates);
            int diffs = 0; for (size_t i = 0; i < std::min(rp.hits.size(), want.hits.size()); ++i) diffs += rp.hits[i] != want.hits[i];
            std::printf("perturbation %s: rates %s (max diff %.3e, limit %.3e), hit differences %d, changed vs baseline: %s\n", kind,
                        exact ? "bit-exact" : "within tolerance", dr, limit, diffs, rp.rates != base.rates ? "yes" : "NO");
            CHECK_MSG(dr <= limit && diffs == 0, "perturbation %s", kind);
            CHECK_MSG(rp.rates != base.rates, "perturbation %s changed nothing (rule H3)", kind);
        }
        // the inject control takes effect at its start frame: identical to the baseline before it
        {
            Controls c(b); Injection inj; inj.neuron = 5; inj.current = 3.0f; inj.startFrame = 100; inj.frames = 20; c.addInjection(inj);
            RunOptions op = o; op.controls = &c; Run rp = runEngine(b, audio, op);
            CHECK(std::equal(rp.rates.begin(), rp.rates.begin() + 100 * b.n, base.rates.begin()));
            CHECK(!std::equal(rp.rates.begin() + 100 * b.n, rp.rates.begin() + 121 * b.n, base.rates.begin() + 100 * b.n));
        }
        // a control change mid-stream: applied at a frame boundary, the frames before it unchanged
        {
            Controls c(b); c.gainScale = 1.3;
            RunOptions op = o; op.controls = &c; op.controlsAtFrame = 150; Run rp = runEngine(b, audio, op);
            CHECK(std::equal(rp.rates.begin(), rp.rates.begin() + 150 * b.n, base.rates.begin()));
            CHECK(rp.rates != base.rates);
        }
    }

    // 6. the runaway guard: a gain sweep 1x to 50x never gives NaN/Inf; the engine recovers after the guard resets
    {
        BrainFile g = b;
        g.runaway.set = true; g.runaway.rateCeiling = 1e3; g.runaway.meanRateLimit = 1.0; g.runaway.frames = 5;
        std::vector<float> loud(audio.size()); for (size_t i = 0; i < loud.size(); ++i) loud[i] = audio[i] * 3.0f;
        int triggered = 0;
        for (double gain : {1.0, 2.0, 5.0, 10.0, 20.0, 50.0}) {
            Controls c(g); c.gainScale = gain;
            for (bool dbl : {true, false}) {
                RunOptions og = o; og.dbl = dbl; og.controls = &c; Run r = runEngine(g, loud, og);
                CHECK_MSG(allFinite(r.rates) && allFinite(r.prob), "gain %.0f %s not finite", gain, dbl ? "double" : "float");
                CHECK_MSG(maxAbs(r.rates) <= 1e3, "gain %.0f: peak %.3g above the ceiling", gain, maxAbs(r.rates));
                triggered += r.runaways;
                if (gain == 50.0 && dbl) std::printf("gain 50x: %d runaway resets, peak rate %.4g\n", r.runaways, maxAbs(r.rates));
            }
        }
        CHECK_MSG(triggered > 0, "the guard never triggered");
        // recovery: after the sweep, a fresh neutral run equals the baseline (the limits leave normal activity alone)
        Run calm = runEngine(g, audio, o);
        CHECK(calm.rates == base.rates && calm.hits == base.hits);
        // and within one engine: trigger, then reset and neutral controls -> baseline
        Engine e(g); EngineConfig cfg; cfg.hostRate = 16000; cfg.doublePrecision = true; cfg.prerollFrames = 0; e.prepare(cfg);
        Controls c(g); c.gainScale = 50.0; e.setControls(c);
        HitEvent ev[256];
        for (size_t p = 0; p + 80 <= loud.size(); p += 80) { const float* ch[1] = {loud.data() + p}; e.process(ch, 1, 80, ev, 256); }
        CHECK(e.runawayCount() > 0);
        Controls n(g); e.setControls(n); e.reset();
        Snapshot s; e.allocateSnapshot(s); std::vector<double> rates;
        for (size_t p = 0; p + 80 <= audio.size(); p += 80) { const float* ch[1] = {audio.data() + p}; e.process(ch, 1, 80, ev, 256); while (e.readSnapshot(s)) rates.insert(rates.end(), s.rates.begin(), s.rates.end()); }
        e.flush(ev, 256); while (e.readSnapshot(s)) rates.insert(rates.end(), s.rates.begin(), s.rates.end());
        CHECK(rates == base.rates);
    }

    // 7. pre-roll: hits muted for N frames after a reset, the network itself untouched
    {
        RunOptions op = o; op.preroll = 100; Run r = runEngine(b, audio, op);
        CHECK(r.rates == base.rates);
        bool mutedOk = true;
        for (size_t f = 0; f < r.frames.size(); ++f) {
            bool muted = f < 100;
            CHECK(r.muted[f] == uint8_t(muted));
            for (int d = 0; d < b.nDrums; ++d) mutedOk = mutedOk && (muted ? r.hits[f * b.nDrums + d] == 0 : r.hits[f * b.nDrums + d] == base.hits[f * b.nDrums + d]);
        }
        CHECK(mutedOk);
    }
    // 8. reset restores the initial state; input trim changes the output
    {
        Engine e(b); EngineConfig cfg; cfg.hostRate = 16000; cfg.doublePrecision = true; cfg.prerollFrames = 0; e.prepare(cfg);
        Snapshot s; e.allocateSnapshot(s); HitEvent ev[256];
        auto run = [&]() { std::vector<double> rates; for (size_t p = 0; p + 80 <= audio.size(); p += 80) { const float* ch[1] = {audio.data() + p}; e.process(ch, 1, 80, ev, 256); while (e.readSnapshot(s)) rates.insert(rates.end(), s.rates.begin(), s.rates.end()); } e.flush(ev, 256); while (e.readSnapshot(s)) rates.insert(rates.end(), s.rates.begin(), s.rates.end()); return rates; };
        std::vector<double> first = run();
        e.reset();
        std::vector<double> second = run();
        CHECK(first == second && first == base.rates);
        e.reset(); e.setInputTrimDb(-6.0);
        CHECK(run() != base.rates);
    }
    return finish();
}
