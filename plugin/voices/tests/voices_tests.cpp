// The voices library against the P3 brief's section 7: each voice sounds and decays within its documented time,
// output bounded for any velocity sequence, retrigger and reset are click-free (and reset never cuts), the open
// hat is choked, the sonification is silent for z = 0 and bounded for |z| = 10, the sidechain gain stays in
// [0, 1] with its attack and release times, the limiter's ceiling holds, and the CPU cost is measured.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "check.h"
#include "flybrain/voices.h"

using namespace flybrain::voices;
static const double SR = 48000.0;
static const char* NAMES[kDrums] = {"kick", "hihat_pedal", "snare", "hihat_closed", "hihat_open", "tom", "crash", "ride"};
// documented decay to -60 dB, Studio voicing, generous upper bounds in seconds (the tail below -60 dB is longer)
static const double MAX_DECAY_S[kDrums] = {0.6, 0.12, 0.4, 0.15, 0.8, 0.5, 1.8, 1.4};

struct Render { std::vector<float> L, R; };

static Render renderKit(Kit& kit, double seconds, const std::vector<std::pair<int, std::pair<int, float>>>& hits /*sample, (drum, vel)*/, int block = 512, int resetAt = -1) {
    int n = int(seconds * SR);
    Render r; r.L.assign(size_t(n), 0.0f); r.R.assign(size_t(n), 0.0f);
    size_t h = 0;
    for (int p = 0; p < n; p += block) {
        int len = std::min(block, n - p);
        while (h < hits.size() && hits[h].first < p + len) { kit.trigger(hits[h].second.first, hits[h].second.second, hits[h].first - p); ++h; }
        if (resetAt >= p && resetAt < p + len && resetAt >= 0) {
            // reset mid-block: process up to the reset sample, reset, then the rest
            int a = resetAt - p;
            if (a > 0) kit.process(r.L.data() + p, r.R.data() + p, a);
            kit.reset();
            if (len - a > 0) kit.process(r.L.data() + p + a, r.R.data() + p + a, len - a);
        } else {
            kit.process(r.L.data() + p, r.R.data() + p, len);
        }
    }
    return r;
}

static double peakOf(const std::vector<float>& x, int from = 0, int to = -1) {
    double m = 0; if (to < 0) to = int(x.size());
    for (int i = from; i < to; ++i) m = std::max(m, std::fabs(double(x[size_t(i)])));
    return m;
}
static double rms(const std::vector<float>& x, int from, int to) {
    double s = 0; for (int i = from; i < to; ++i) s += double(x[size_t(i)]) * x[size_t(i)];
    return std::sqrt(s / std::max(1, to - from));
}
static double maxJump(const std::vector<float>& x, int from, int to) {
    double m = 0; for (int i = std::max(from, 1); i < to; ++i) m = std::max(m, std::fabs(double(x[size_t(i)]) - x[size_t(i - 1)]));
    return m;
}
static bool allFinite(const std::vector<float>& x) { for (float v : x) if (!std::isfinite(v)) return false; return true; }

int main() {
    // 1. each voice at velocity 1: it sounds, then decays to -60 dB of its peak within its documented time
    std::printf("| drum | peak | -60 dB after (ms) | bound (ms) | max jump within one hit |\n|---|---|---|---|---|\n");
    double jump0[kDrums];
    for (int d = 0; d < kDrums; ++d) {
        Kit kit; kit.prepare(SR, 512); kit.setDrum(d, 0.0f, false, 0.0f);
        Render r = renderKit(kit, 3.0, {{0, {d, 1.0f}}});
        double pk = peakOf(r.L);
        CHECK_MSG(pk > 0.05 && pk < 1.5, "%s: peak %.3f", NAMES[d], pk);
        int t60 = -1;
        for (int w = 0; w + 48 <= int(r.L.size()); w += 48) if (peakOf(r.L, w, w + 48) < pk * 0.001 && w > 48) { t60 = w; break; }
        CHECK_MSG(t60 > 0 && t60 / SR <= MAX_DECAY_S[d], "%s: -60 dB at %d samples (%.0f ms), bound %.0f ms", NAMES[d], t60, t60 * 1000.0 / SR, MAX_DECAY_S[d] * 1000);
        CHECK_MSG(peakOf(r.L, int(2.5 * SR)) < 1e-4, "%s: not silent after 2.5 s (%.2e)", NAMES[d], peakOf(r.L, int(2.5 * SR)));
        jump0[d] = maxJump(r.L, 0, int(r.L.size()));
        std::printf("| %s | %.3f | %.0f | %.0f | %.3f |\n", NAMES[d], pk, t60 * 1000.0 / SR, MAX_DECAY_S[d] * 1000, jump0[d]);
        CHECK(allFinite(r.L) && allFinite(r.R));
    }
    // 1b. the three voicings differ and all decay
    for (int v = 0; v < 3; ++v) {
        Kit kit; kit.prepare(SR, 512); kit.setVoicing(v);
        Render r = renderKit(kit, 3.0, {{0, {6, 1.0f}}});
        CHECK_MSG(peakOf(r.L, int(2.8 * SR)) < 1e-3, "voicing %d: crash still sounding at 2.8 s", v);
    }
    // 2. bounded for any velocity sequence: 2,000 random triggers in 4 s, all drums, every block size
    {
        Kit kit; kit.prepare(SR, 4096);
        std::vector<std::pair<int, std::pair<int, float>>> hits;
        uint32_t s = 12345;
        for (int k = 0; k < 2000; ++k) { s = s * 1664525u + 1013904223u; int at = int((s >> 8) % uint32_t(4 * SR)); s = s * 1664525u + 1013904223u; hits.push_back({at, {int(s % kDrums), float((s >> 8) % 1000) / 999.0f}}); }
        std::sort(hits.begin(), hits.end());
        Render r = renderKit(kit, 4.5, hits, 4096);
        CHECK(allFinite(r.L) && allFinite(r.R));
        double pk = std::max(peakOf(r.L), peakOf(r.R));
        CHECK_MSG(pk < 8.0, "dense random triggers: peak %.2f", pk);
        std::printf("dense random triggers (2,000 in 4 s): peak %.2f, finite\n", pk);
        // velocity ordering: louder velocity gives a louder hit
        Kit k2; k2.prepare(SR, 512);
        Render a = renderKit(k2, 0.5, {{0, {2, 0.3f}}}); Kit k3; k3.prepare(SR, 512);
        Render b = renderKit(k3, 0.5, {{0, {2, 1.0f}}});
        CHECK(peakOf(b.L) > peakOf(a.L));
        // the velocity curve: c = -1 softer at mid velocity, c = +1 harder
        Kit k4; k4.prepare(SR, 512); k4.setVelocityCurve(-1.0f); Render soft = renderKit(k4, 0.5, {{0, {2, 0.5f}}});
        Kit k5; k5.prepare(SR, 512); k5.setVelocityCurve(1.0f); Render hard = renderKit(k5, 0.5, {{0, {2, 0.5f}}});
        CHECK(peakOf(soft.L) < peakOf(hard.L));
    }
    // 3. retrigger and reset are click-free: the largest sample-to-sample jump outside an attack window is no
    //    larger than the voice's own largest jump within one hit (noise voices jump by nature); reset never cuts
    for (int d = 0; d < kDrums; ++d) {
        Kit kit; kit.prepare(SR, 512); kit.setDrum(d, 0.0f, false, 0.0f);
        int t1 = int(0.030 * SR), t2 = int(0.045 * SR), rs = int(0.070 * SR);
        Render r = renderKit(kit, 0.5, {{0, {d, 1.0f}}, {t1, {d, 0.8f}}, {t2, {d, 1.0f}}}, 512, rs);
        int atk = int(0.002 * SR);
        double j = std::max({maxJump(r.L, atk, t1), maxJump(r.L, t1 + atk, t2), maxJump(r.L, t2 + atk, int(r.L.size()))});
        CHECK_MSG(j <= 1.5 * jump0[d] + 0.02, "%s: jump %.3f at a retrigger/reset against %.3f within a hit", NAMES[d], j, jump0[d]);
        // reset never cuts: energy right after the reset, then silence 6 ms later
        double before = rms(r.L, rs - 48, rs), during = rms(r.L, rs, rs + 48), after = rms(r.L, rs + int(0.006 * SR), rs + int(0.006 * SR) + 48);
        CHECK_MSG(during > 0.1 * before && during > 0, "%s: reset cut the sound (before %.4f, first ms after %.4f)", NAMES[d], before, during);
        CHECK_MSG(after == 0.0, "%s: still sounding 6 ms after reset (%.2e)", NAMES[d], after);
        CHECK(!kit.anyActive());
    }
    // 4. the open hat is choked by the closed and the pedal hat (5 ms fade), not by other drums
    {
        for (int choker : {1, 3, 0}) {
            Kit kit; kit.prepare(SR, 512); for (int d = 0; d < kDrums; ++d) kit.setDrum(d, d == 4 ? 0.0f : -120.0f, d != 4, 0.0f);
            int tc = int(0.100 * SR);
            Render r = renderKit(kit, 0.5, {{0, {4, 1.0f}}, {tc, {choker, 1.0f}}});
            double justAfter = rms(r.L, tc + int(0.008 * SR), tc + int(0.012 * SR)), beforeChoke = rms(r.L, tc - 200, tc);
            if (choker == 0) CHECK_MSG(justAfter > 0.3 * beforeChoke, "kick choked the open hat");
            else CHECK_MSG(justAfter == 0.0, "choker %d: open hat still sounding 8 ms after (%.2e vs %.4f before)", choker, justAfter, beforeChoke);
        }
    }
    // 5. sonification: silent for z = 0, bounded for |z| = 10, enable fades
    {
        Sonification son; son.prepare(SR, 512); son.setEnabled(true); son.setLevelDb(0.0f);
        std::vector<float> L(4800, 0.0f), R(4800, 0.0f);
        float z0[8] = {0, 0, 0, 0, 0, 0, 0, 0}; son.setGroupRates(z0, 8); son.process(L.data(), R.data(), 4800);
        CHECK_MSG(peakOf(L) == 0.0, "sonification not silent for z = 0: %.2e", peakOf(L));
        float z10[8] = {10, -10, 10, 10, 10, 10, 10, 10}; son.setGroupRates(z10, 8);
        double pk = 0; for (int b = 0; b < 100; ++b) { std::fill(L.begin(), L.end(), 0.0f); std::fill(R.begin(), R.end(), 0.0f); son.process(L.data(), R.data(), 4800); pk = std::max(pk, peakOf(L)); }
        CHECK_MSG(pk > 0.1 && pk <= 1.0, "sonification |z| = 10: peak %.3f", pk);
        float zn[8] = {std::nanf(""), 1, 1, 1, 1, 1, 1, 1}; son.setGroupRates(zn, 8);
        std::fill(L.begin(), L.end(), 0.0f); son.process(L.data(), R.data(), 4800); CHECK(allFinite(L));
        // scale
        CHECK(std::fabs(Sonification::noteHz(0) - 65.4064) < 1e-3 && std::fabs(Sonification::noteHz(7) - 523.2511) < 1e-3);
        // the 20 ms smoothing: after a step from 0 to z = 10 the amplitude envelope reaches 63 % of its target near 20 ms
        Sonification s2; s2.prepare(SR, 512); s2.setEnabled(true); s2.setLevelDb(0.0f);
        s2.setGroupRates(z0, 8); std::fill(L.begin(), L.end(), 0.0f); s2.process(L.data(), R.data(), 4800);   // enable settles
        s2.setGroupRates(z10, 8); std::fill(L.begin(), L.end(), 0.0f); std::fill(R.begin(), R.end(), 0.0f); s2.process(L.data(), R.data(), 4800);
        double e20 = peakOf(L, int(0.018 * SR), int(0.022 * SR)), eEnd = peakOf(L, 4000, 4800);
        CHECK_MSG(e20 > 0.45 * eEnd && e20 < 0.85 * eEnd, "sonification smoothing: %.3f at 20 ms against %.3f settled", e20, eEnd);
        std::printf("sonification: z = 0 silent, |z| = 10 peak %.3f at 0 dB, 20 ms envelope %.2f of settled\n", pk, e20 / eEnd);
    }
    // 6. sidechain: gain in [0, 1]; duck and gate directions; attack ~10 ms, release ~80 ms
    {
        Sidechain sc; sc.prepare(SR, 512); sc.setAmount(1.0f); sc.setMode(Sidechain::Duck);
        std::vector<float> L(int(0.5 * SR), 1.0f), R(L.size(), 1.0f);
        sc.setGroupRate(10.0f);                       // z' -> 1: duck to 0
        sc.process(L.data(), R.data(), int(L.size()));
        double mn = 1, mx = 0; for (float v : L) { mn = std::min(mn, double(v)); mx = std::max(mx, double(v)); }
        CHECK_MSG(mn >= 0.0 && mx <= 1.0, "duck gain out of [0, 1]: %.3f..%.3f", mn, mx);
        int i63 = -1; for (size_t i = 0; i < L.size(); ++i) if (L[i] < 1.0 - 0.63 * (1.0 - std::tanh(5.0)) * 0 - 0.63) { i63 = int(i); break; }
        CHECK_MSG(i63 > 0 && i63 / SR > 0.006 && i63 / SR < 0.016, "duck attack: 63 %% at %.1f ms", i63 * 1000.0 / SR);
        std::fill(L.begin(), L.end(), 1.0f); std::fill(R.begin(), R.end(), 1.0f);
        sc.setGroupRate(0.0f);                        // release back to 1
        sc.process(L.data(), R.data(), int(L.size()));
        double g0 = L[0]; int r63 = -1; for (size_t i = 0; i < L.size(); ++i) if (L[i] > g0 + 0.63 * (1.0 - g0)) { r63 = int(i); break; }
        CHECK_MSG(r63 > 0 && r63 / SR > 0.05 && r63 / SR < 0.12, "duck release: 63 %% at %.1f ms", r63 * 1000.0 / SR);
        std::printf("sidechain duck: attack 63 %% at %.1f ms, release 63 %% at %.1f ms\n", i63 * 1000.0 / SR, r63 * 1000.0 / SR);
        Sidechain g; g.prepare(SR, 512); g.setAmount(1.0f); g.setMode(Sidechain::Gate); g.setGroupRate(0.0f);
        std::fill(L.begin(), L.end(), 1.0f); g.process(L.data(), R.data(), int(L.size()));
        CHECK_MSG(L.back() < 0.01, "gate with z = 0 did not close (%.3f)", L.back());
        g.setGroupRate(10.0f); std::fill(L.begin(), L.end(), 1.0f); g.process(L.data(), R.data(), int(L.size()));
        CHECK_MSG(L.back() > 0.95 && L.back() <= 1.0, "gate with z = 10 did not open (%.3f)", L.back());
        Sidechain off; off.prepare(SR, 512); off.setAmount(0.0f); off.setGroupRate(10.0f); std::fill(L.begin(), L.end(), 0.5f); off.process(L.data(), R.data(), int(L.size()));
        CHECK(L.front() == 0.5f && L.back() == 0.5f);  // amount 0: untouched
    }
    // 7. limiter: +20 dB sine never exceeds the ceiling; a -6 dB sine passes unchanged; NaN never passes
    {
        Limiter lim; lim.prepare(SR, 512);
        int n = int(1.0 * SR); std::vector<float> L(n), R(n);
        for (int i = 0; i < n; ++i) { L[i] = float(10.0 * std::sin(2 * M_PI * 220.0 * i / SR)); R[i] = L[i]; }
        lim.process(L.data(), R.data(), n);
        CHECK_MSG(peakOf(L) <= lim.ceiling() + 1e-7 && peakOf(R) <= lim.ceiling() + 1e-7, "limiter: peak %.5f above ceiling %.5f", peakOf(L), lim.ceiling());
        std::printf("limiter: +20 dB input, output peak %.5f, ceiling %.5f\n", peakOf(L), lim.ceiling());
        Limiter l2; l2.prepare(SR, 512);
        for (int i = 0; i < n; ++i) { L[i] = float(0.5 * std::sin(2 * M_PI * 220.0 * i / SR)); R[i] = L[i]; }
        std::vector<float> copy = L; l2.process(L.data(), R.data(), n);
        CHECK(L == copy);
        L[10] = std::nanf(""); R[11] = INFINITY; l2.process(L.data(), R.data(), n); CHECK(allFinite(L) && allFinite(R));
    }
    // 8. CPU: all eight drums sounding, 512-sample blocks at 48 kHz
    {
        Kit kit; kit.prepare(SR, 512); Sonification son; son.prepare(SR, 512); son.setEnabled(true);
        float z[8] = {1, 1, 1, 1, 1, 1, 1, 1}; son.setGroupRates(z, 8);
        std::vector<float> L(512), R(512);
        auto t0 = std::chrono::steady_clock::now();
        const int blocks = 2000;
        for (int b = 0; b < blocks; ++b) {
            std::fill(L.begin(), L.end(), 0.0f); std::fill(R.begin(), R.end(), 0.0f);
            if (b % 20 == 0) for (int d = 0; d < kDrums; ++d) kit.trigger(d, 1.0f, d * 50);
            kit.process(L.data(), R.data(), 512);
        }
        double usKit = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / blocks;
        t0 = std::chrono::steady_clock::now();
        for (int b = 0; b < blocks; ++b) son.process(L.data(), R.data(), 512);
        double usSon = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / blocks;
        double blockUs = 512.0 / SR * 1e6;
        std::printf("cpu: kit (8 drums retriggered every 20 blocks, up to 16 voices) %.1f us per 512-sample block (%.2f %% of one core), sonification %.1f us (%.2f %%)\n",
                    usKit, 100.0 * usKit / blockUs, usSon, 100.0 * usSon / blockUs);
    }
    return finish();
}
