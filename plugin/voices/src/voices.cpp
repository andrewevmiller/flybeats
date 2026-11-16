#include "flybrain/voices.h"

#include <algorithm>
#include <cmath>

namespace flybrain {
namespace voices {

namespace {
constexpr double kPi = 3.14159265358979323846;

inline double dbToLin(float db) { return db <= -120.0f ? 0.0 : std::pow(10.0, double(db) / 20.0); }
inline uint32_t xorshift(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
inline double noise(uint32_t& s) { return double(xorshift(s)) * (2.0 / 4294967296.0) - 1.0; }
inline double decayCoefFor(double seconds, double sr) { return seconds <= 0 ? 0.0 : std::exp(-6.9077553 / (seconds * sr)); }   // -60 dB in `seconds`
inline double onePoleLp(double hz, double sr) { return 1.0 - std::exp(-2.0 * kPi * hz / sr); }
inline double onePoleHp(double hz, double sr) { return std::exp(-2.0 * kPi * hz / sr); }

// RBJ band-pass (constant skirt gain), coefficients b0 b1 b2 a1 a2 normalised by a0
void bandpass(double hz, double q, double sr, double* c) {
    double w = 2.0 * kPi * hz / sr, s = std::sin(w), co = std::cos(w), alpha = s / (2.0 * q);
    double a0 = 1.0 + alpha;
    c[0] = (q * alpha) / a0; c[1] = 0.0; c[2] = (-q * alpha) / a0; c[3] = (-2.0 * co) / a0; c[4] = (1.0 - alpha) / a0;
}
inline double biquad(double x, const double* c, double* st) {   // st: x1 x2 y1 y2
    double y = c[0] * x + c[1] * st[0] + c[2] * st[1] - c[3] * st[2] - c[4] * st[3];
    st[1] = st[0]; st[0] = x; st[3] = st[2]; st[2] = y;
    return y;
}

// Voicing table: decay seconds per drum for Studio, Tight, Room; the tonal sweep and filter settings follow.
struct DrumSpec { double decay[3]; double decay2[3]; };
const DrumSpec kSpec[kDrums] = {
    {{0.35, 0.20, 0.50}, {0.004, 0.003, 0.006}},   // kick: body, click
    {{0.040, 0.025, 0.060}, {0, 0, 0}},           // pedal hat
    {{0.12, 0.08, 0.18}, {0.18, 0.12, 0.26}},      // snare: tone, noise
    {{0.060, 0.040, 0.090}, {0, 0, 0}},           // closed hat
    {{0.45, 0.30, 0.70}, {0, 0, 0}},              // open hat
    {{0.30, 0.18, 0.45}, {0, 0, 0}},              // tom
    {{1.2, 0.8, 1.8}, {0, 0, 0}},                 // crash
    {{0.9, 0.6, 1.3}, {0, 0, 0}},                 // ride
};
}  // namespace

// ------------------------------------------------------------------------------------------------ Kit
Kit::Kit() {
    for (int d = 0; d < kDrums; ++d) { gainLin_[d] = 1.0f; panL_[d] = panR_[d] = 0.70710677f; mute_[d] = false; nextSlot_[d] = 0; }
    // a default stereo spread, equal power
    const float pans[kDrums] = {0.0f, -0.3f, 0.1f, 0.35f, 0.4f, -0.25f, -0.5f, 0.55f};
    for (int d = 0; d < kDrums; ++d) setDrum(d, 0.0f, false, pans[d]);
    uint32_t seed = 0x9E3779B9u;
    for (int d = 0; d < kDrums; ++d) for (int s = 0; s < kSlotsPerDrum; ++s) voices_[d][s].rng = seed += 0x7F4A7C15u;
}

void Kit::prepare(double sampleRate, int maxBlock) {
    sr_ = sampleRate > 1000 ? sampleRate : 48000.0;
    maxBlock_ = maxBlock;
    for (int d = 0; d < kDrums; ++d) for (int s = 0; s < kSlotsPerDrum; ++s) { Voice& v = voices_[d][s]; v.active = false; v.fading = false; v.fade = 1; }
    nQueued_ = 0;
}

void Kit::reset() {
    nQueued_ = 0;
    const double step = 1.0 / (0.005 * sr_);          // 5 ms
    for (int d = 0; d < kDrums; ++d) for (int s = 0; s < kSlotsPerDrum; ++s) {
        Voice& v = voices_[d][s];
        if (v.active && (!v.fading || v.fadeStep < step)) { v.fading = true; v.fadeStep = step; }
    }
}

void Kit::setVoicing(int voicing) { voicing_ = voicing < 0 ? 0 : voicing > 2 ? 2 : voicing; }

void Kit::setDrum(int drum, float gainDb, bool mute, float pan) {
    if (drum < 0 || drum >= kDrums) return;
    gainLin_[drum] = float(dbToLin(gainDb));
    mute_[drum] = mute;
    pan = pan < -1 ? -1.0f : pan > 1 ? 1.0f : pan;
    double a = (double(pan) + 1.0) * 0.25 * kPi;       // equal power
    panL_[drum] = float(std::cos(a));
    panR_[drum] = float(std::sin(a));
}

void Kit::setVelocityCurve(float c) { curve_ = c < -1 ? -1.0f : c > 1 ? 1.0f : c; }

void Kit::trigger(int drum, float velocity01, int sampleOffset) {
    if (drum < 0 || drum >= kDrums || nQueued_ >= kMaxTriggers) return;
    queue_[nQueued_++] = Trigger{drum, velocity01 < 0 ? 0.0f : velocity01 > 1 ? 1.0f : velocity01, sampleOffset};
}

bool Kit::anyActive() const {
    for (int d = 0; d < kDrums; ++d) for (int s = 0; s < kSlotsPerDrum; ++s) if (voices_[d][s].active) return true;
    return false;
}

void Kit::chokeOpenHat() {
    const double step = 1.0 / (0.005 * sr_);
    for (int s = 0; s < kSlotsPerDrum; ++s) {
        Voice& v = voices_[4][s];
        if (v.active && (!v.fading || v.fadeStep < step)) { v.fading = true; v.fadeStep = step; }
    }
}

void Kit::start(Voice& v, int drum, float vel) {
    const double sr = sr_;
    const DrumSpec& sp = kSpec[drum];
    double vc = std::pow(double(vel), std::pow(2.0, -double(curve_)));    // the velocity curve
    double bright = 0.5 + 0.5 * vc;                                        // velocity -> brightness
    uint32_t rng = v.rng;
    v = Voice();
    v.rng = rng;
    v.active = true; v.drum = drum; v.amp = vc; v.env = 1; v.env2 = 1; v.fade = 1; v.fading = false; v.age = 0;
    v.decayCoef = decayCoefFor(sp.decay[voicing_], sr);
    v.decayCoef2 = decayCoefFor(sp.decay2[voicing_], sr);
    v.hpA = onePoleHp(20.0, sr); v.lpA = onePoleLp(20000.0 * bright, sr);
    switch (drum) {
    case 0:   // kick: sine sweep 150 -> 45 Hz over ~60 ms, plus a 4 ms click of noise
        v.freq = 150.0; v.freqEnd = 45.0; v.sweepCoef = std::exp(std::log(45.0 / 150.0) / (0.06 * sr));
        v.toneMix = 1.0; v.clickMix = 0.5 * bright; v.lpA = onePoleLp(2500.0 * bright + 500.0, sr);
        break;
    case 1:   // pedal hat: short dark noise
        v.noiseMix = 0.6; v.hpA = onePoleHp(4000.0, sr); v.lpA = onePoleLp(9000.0 * bright + 2000.0, sr);
        break;
    case 2:   // snare: 180 + 330 Hz tone, plus high-passed noise
        v.freq = 180.0; v.freqEnd = 180.0; v.sweepCoef = 1.0; v.toneMix = 0.55; v.noiseMix = 0.75;
        v.hpA = onePoleHp(1500.0, sr); v.lpA = onePoleLp(12000.0 * bright + 3000.0, sr);
        break;
    case 3:   // closed hat: short bright noise
        v.noiseMix = 0.7; v.hpA = onePoleHp(6000.0, sr); v.lpA = onePoleLp(14000.0 * bright + 3000.0, sr);
        break;
    case 4:   // open hat: long noise
        v.noiseMix = 0.6; v.hpA = onePoleHp(5000.0, sr); v.lpA = onePoleLp(14000.0 * bright + 3000.0, sr);
        break;
    case 5:   // tom: sine sweep 180 -> 90 Hz over 100 ms
        v.freq = 180.0; v.freqEnd = 90.0; v.sweepCoef = std::exp(std::log(90.0 / 180.0) / (0.10 * sr));
        v.toneMix = 1.0; v.clickMix = 0.15 * bright; v.decayCoef2 = decayCoefFor(0.003, sr); v.lpA = onePoleLp(3000.0 * bright + 800.0, sr);
        break;
    case 6:   // crash: noise through two band resonances (3.5 kHz, 6 kHz) plus a little raw noise
        v.noiseMix = 0.5; v.hpA = onePoleHp(2500.0, sr); v.lpA = onePoleLp(16000.0 * bright + 2000.0, sr);
        bandpass(3500.0, 2.0, sr, v.bp1c); bandpass(6000.0, 2.5, sr, v.bp2c);
        break;
    case 7:   // ride: noise plus three metallic partials (2.1, 3.3, 4.9 kHz)
        v.noiseMix = 0.35; v.toneMix = 0.4; v.freq = 2100.0; v.freqEnd = 2100.0; v.sweepCoef = 1.0;
        v.hpA = onePoleHp(3000.0, sr); v.lpA = onePoleLp(15000.0 * bright + 2000.0, sr);
        bandpass(4200.0, 3.0, sr, v.bp1c);
        break;
    default: break;
    }
}

float Kit::render(Voice& v) {
    double out = 0.0;
    const double sr = sr_;
    if (v.toneMix > 0) {
        double tone;
        if (v.drum == 2) {                                   // snare: two partials
            tone = 0.7 * std::sin(v.phase) + 0.3 * std::sin(v.p2);
            v.p2 += 2.0 * kPi * 330.0 / sr;
        } else if (v.drum == 7) {                            // ride: three inharmonic partials
            tone = 0.5 * std::sin(v.phase) + 0.3 * std::sin(v.p2) + 0.2 * std::sin(v.p3);
            v.p2 += 2.0 * kPi * 3300.0 / sr; v.p3 += 2.0 * kPi * 4900.0 / sr;
        } else {
            tone = std::sin(v.phase);
        }
        v.phase += 2.0 * kPi * v.freq / sr;
        if (v.phase > 2.0 * kPi) v.phase -= 2.0 * kPi;
        if (v.p2 > 2.0 * kPi) v.p2 -= 2.0 * kPi;
        if (v.p3 > 2.0 * kPi) v.p3 -= 2.0 * kPi;
        if (v.freq > v.freqEnd) { v.freq *= v.sweepCoef; if (v.freq < v.freqEnd) v.freq = v.freqEnd; }
        out += v.toneMix * tone * v.env;
    }
    if (v.noiseMix > 0 || v.clickMix > 0) {
        double n = noise(v.rng);
        if (v.noiseMix > 0) {
            double e = (v.drum == 2) ? v.env2 : v.env;        // snare's noise has its own decay
            double f = n;
            if (v.drum == 6) f = 0.5 * biquad(n, v.bp1c, v.bp1) + 0.5 * biquad(n, v.bp2c, v.bp2) + 0.2 * n;
            else if (v.drum == 7) f = 0.6 * biquad(n, v.bp1c, v.bp1) + 0.4 * n;
            out += v.noiseMix * f * e;
        }
        if (v.clickMix > 0) out += v.clickMix * n * v.env2;
    }
    // one-pole high-pass then low-pass (brightness)
    double hp = v.hpA * (v.hpY + out - v.hpX); v.hpX = out; v.hpY = hp;
    v.lpY += v.lpA * (hp - v.lpY);
    out = v.lpY * v.amp;
    v.env *= v.decayCoef;
    v.env2 *= v.decayCoef2;
    if (v.fading) { v.fade -= v.fadeStep; if (v.fade <= 0) { v.fade = 0; v.active = false; } }
    out *= v.fade;
    v.age += 1;
    if (v.env < 1e-5 && v.env2 < 1e-5) v.active = false;     // -100 dB: done
    return float(out);
}

void Kit::process(float* L, float* R, int n) {
    // sort the queue by offset (insertion sort, at most kMaxTriggers entries, no allocation)
    for (int i = 1; i < nQueued_; ++i) {
        Trigger t = queue_[i]; int j = i - 1;
        while (j >= 0 && queue_[j].offset > t.offset) { queue_[j + 1] = queue_[j]; --j; }
        queue_[j + 1] = t;
    }
    int next = 0;
    const double retrigStep = 1.0 / (0.001 * sr_);            // 1 ms
    for (int i = 0; i < n; ++i) {
        while (next < nQueued_ && std::min(std::max(queue_[next].offset, 0), n - 1) <= i) {
            const Trigger& t = queue_[next++];
            int slot = nextSlot_[t.drum];
            Voice& old = voices_[t.drum][slot ^ 1];
            if (old.active && (!old.fading || old.fadeStep < retrigStep)) { old.fading = true; old.fadeStep = retrigStep; }
            start(voices_[t.drum][slot], t.drum, t.vel);
            nextSlot_[t.drum] = slot ^ 1;
            if (t.drum == 1 || t.drum == 3) chokeOpenHat();
        }
        float l = 0.0f, r = 0.0f;
        for (int d = 0; d < kDrums; ++d) {
            float g = mute_[d] ? 0.0f : gainLin_[d];
            for (int s = 0; s < kSlotsPerDrum; ++s) {
                Voice& v = voices_[d][s];
                if (!v.active) continue;
                float x = render(v) * g;
                l += x * panL_[d]; r += x * panR_[d];
            }
        }
        L[i] += l; R[i] += r;
    }
    nQueued_ = 0;
}

// ------------------------------------------------------------------------------------------------ Sonification
double Sonification::noteHz(int group) {
    static const double hz[kDrums] = {65.4064, 97.9989, 130.8128, 195.9977, 261.6256, 329.6276, 391.9954, 523.2511};
    return (group >= 0 && group < kDrums) ? hz[group] : 440.0;
}

Sonification::Sonification() { for (int g = 0; g < kDrums; ++g) { phase_[g] = 0; inc_[g] = 0; target_[g] = 0; amp_[g] = 0; } }

void Sonification::prepare(double sampleRate, int) {
    sr_ = sampleRate > 1000 ? sampleRate : 48000.0;
    smooth_ = onePoleLp(1.0 / (2.0 * kPi * 0.020), sr_);       // 20 ms time constant
    for (int g = 0; g < kDrums; ++g) inc_[g] = 2.0 * kPi * noteHz(g) / sr_;
    reset();
}

void Sonification::reset() { for (int g = 0; g < kDrums; ++g) { phase_[g] = 0; amp_[g] = 0; target_[g] = 0; } enable_ = enableTarget_; }
void Sonification::setLevelDb(float db) { level_ = dbToLin(db); }
void Sonification::setEnabled(bool on) { enableTarget_ = on ? 1.0 : 0.0; }

void Sonification::setGroupRates(const float* z, int nGroups) {
    for (int g = 0; g < kDrums && g < nGroups; ++g) {
        double x = double(z[g]);
        if (!(x > 0)) x = 0;                                     // NaN or negative: silent
        target_[g] = std::tanh(x / 3.0);
    }
}

void Sonification::process(float* L, float* R, int n) {
    if (enable_ <= 0 && enableTarget_ <= 0) { bool quiet = true; for (int g = 0; g < kDrums; ++g) quiet = quiet && amp_[g] < 1e-7; if (quiet) return; }
    for (int i = 0; i < n; ++i) {
        enable_ += smooth_ * (enableTarget_ - enable_);
        double s = 0.0;
        for (int g = 0; g < kDrums; ++g) {
            amp_[g] += smooth_ * (target_[g] - amp_[g]);
            if (amp_[g] > 1e-7) s += amp_[g] * std::sin(phase_[g]);
            phase_[g] += inc_[g];
            if (phase_[g] > 2.0 * kPi) phase_[g] -= 2.0 * kPi;
        }
        float x = float(s * (level_ / double(kDrums)) * enable_);   // eight voices at full amplitude sum to `level`
        L[i] += x; R[i] += x;
    }
}

// ------------------------------------------------------------------------------------------------ Sidechain
Sidechain::Sidechain() {}
void Sidechain::prepare(double sampleRate, int) {
    sr_ = sampleRate > 1000 ? sampleRate : 48000.0;
    attack_ = onePoleLp(1.0 / (2.0 * kPi * 0.010), sr_);
    release_ = onePoleLp(1.0 / (2.0 * kPi * 0.080), sr_);
    reset();
}
void Sidechain::reset() { gain_ = 1; target_ = 1; z_ = 0; }
void Sidechain::setAmount(float a) { amount_ = a < 0 ? 0.0 : a > 1 ? 1.0 : double(a); }
void Sidechain::setMode(int m) { mode_ = m == Gate ? Gate : Duck; }
void Sidechain::setGroupRate(float z) {
    double x = double(z);
    if (!(x > 0)) x = 0;
    z_ = std::tanh(x / 2.0);
    target_ = mode_ == Duck ? 1.0 - amount_ * z_ : (amount_ <= 0 ? 1.0 : amount_ * z_);
    target_ = target_ < 0 ? 0.0 : target_ > 1 ? 1.0 : target_;
}
void Sidechain::process(float* L, float* R, int n) {
    for (int i = 0; i < n; ++i) {
        gain_ += (target_ < gain_ ? attack_ : release_) * (target_ - gain_);
        if (gain_ < 0) gain_ = 0; else if (gain_ > 1) gain_ = 1;
        L[i] = float(L[i] * gain_); R[i] = float(R[i] * gain_);
    }
}

// ------------------------------------------------------------------------------------------------ Limiter
Limiter::Limiter() {}
void Limiter::prepare(double sampleRate, int) {
    sr_ = sampleRate > 1000 ? sampleRate : 48000.0;
    attack_ = onePoleLp(1.0 / (2.0 * kPi * 0.001), sr_);
    release_ = onePoleLp(1.0 / (2.0 * kPi * 0.050), sr_);
    ceiling_ = std::pow(10.0, -0.3 / 20.0);
    reset();
}
void Limiter::reset() { gain_ = 1; }
void Limiter::process(float* L, float* R, int n) {
    const float c = float(ceiling_);
    for (int i = 0; i < n; ++i) {
        double peak = std::max(std::fabs(double(L[i])), std::fabs(double(R[i])));
        if (!std::isfinite(peak)) { L[i] = 0; R[i] = 0; continue; }           // never pass NaN or Inf
        double want = peak > ceiling_ ? ceiling_ / peak : 1.0;
        gain_ += (want < gain_ ? attack_ : release_) * (want - gain_);
        float l = float(L[i] * gain_), r = float(R[i] * gain_);
        L[i] = l > c ? c : l < -c ? -c : l;                                   // the guarantee
        R[i] = r > c ? c : r < -c ? -c : r;
    }
}

}  // namespace voices
}  // namespace flybrain
