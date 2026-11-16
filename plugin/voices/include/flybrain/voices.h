// flybrain_voices: the synthesized 8-piece kit, the Motor Sonification voices, Fly Sidechain and the output
// limiter. JUCE-free, no third-party dependency. Every class is allocation-free after prepare() and never throws.
// Drum order is the brain's pieces: 0 kick, 1 hihat_pedal, 2 snare, 3 hihat_closed, 4 hihat_open, 5 tom,
// 6 crash, 7 ride. No samples are used anywhere: every sound is synthesized (licence-clean by construction).
#pragma once
#include <cstdint>

namespace flybrain {
namespace voices {

constexpr int kDrums = 8;
constexpr int kMaxTriggers = 64;      // queued triggers per block
constexpr int kSlotsPerDrum = 2;      // retrigger starts the other slot; the old one fades out over 1 ms

// ------------------------------------------------------------------------------------------------ Kit
class Kit {
public:
    Kit();
    void prepare(double sampleRate, int maxBlock);
    void reset();                                   // every sounding voice fades out over 5 ms, never a cut
    void setVoicing(int voicing);                   // 0 Studio, 1 Tight, 2 Room
    void setDrum(int drum, float gainDb, bool mute, float pan);   // pan -1 (left) .. +1 (right)
    void setVelocityCurve(float c);                 // v' = v ^ (2 ^ -c); c < 0 softer, c > 0 harder
    void trigger(int drum, float velocity01, int sampleOffset);   // queued for the next process(); offset clamped to the block
    void process(float* L, float* R, int n);        // adds into L and R
    bool anyActive() const;
    int voicing() const { return voicing_; }
    double sampleRate() const { return sr_; }

    struct Voice {                                  // one sounding instance of a drum
        bool active = false;
        int drum = 0;
        double phase = 0, freq = 0, freqEnd = 0, sweepCoef = 1;    // tonal part
        double p2 = 0, p3 = 0, p4 = 0;                              // extra partials (ride, snare)
        double amp = 0, env = 1, decayCoef = 1;                     // amplitude envelope (exponential)
        double env2 = 1, decayCoef2 = 1;                            // second envelope (noise or click)
        double fade = 1, fadeStep = 0;                              // 1 ms / 5 ms fade-outs (retrigger, reset, choke)
        bool fading = false;
        uint32_t rng = 1;
        // filters: one-pole highpass and lowpass, two RBJ bandpasses
        double hpY = 0, hpX = 0, hpA = 0, lpY = 0, lpA = 0;
        double bp1[4] = {0, 0, 0, 0}, bp2[4] = {0, 0, 0, 0};
        double bp1c[5] = {0, 0, 0, 0, 0}, bp2c[5] = {0, 0, 0, 0, 0};
        double noiseMix = 0, toneMix = 0, clickMix = 0;
        double age = 0;                                             // samples since trigger
    };

private:
    struct Trigger { int drum; float vel; int offset; };
    void start(Voice& v, int drum, float vel);
    float render(Voice& v);                         // one mono sample, advances the voice
    void chokeOpenHat();
    double sr_ = 48000;
    int maxBlock_ = 512;
    int voicing_ = 0;
    float curve_ = 0;
    float gainLin_[kDrums], panL_[kDrums], panR_[kDrums];
    bool mute_[kDrums];
    Voice voices_[kDrums][kSlotsPerDrum];
    int nextSlot_[kDrums];
    Trigger queue_[kMaxTriggers];
    int nQueued_ = 0;
};

// ------------------------------------------------------------------------------------------------ Sonification
// Eight continuous voices, one per motor group, on a documented scale: kick C2, pedal hat G2, snare C3, closed
// hat G3, open hat C4, tom E4, crash G4, ride C5. Amplitude = level x tanh(max(z, 0) / 3), smoothed over 20 ms.
// Silent for z <= 0, bounded for any z (tanh < 1). This is a sonification of the network's activity, not music
// the fly plays: it is labelled as such in the UI.
class Sonification {
public:
    Sonification();
    void prepare(double sampleRate, int maxBlock);
    void reset();
    void setLevelDb(float db);
    void setEnabled(bool on);                       // fades in or out over 20 ms, no click
    void setGroupRates(const float* z, int nGroups); // once per 5 ms frame; nGroups <= 8
    void process(float* L, float* R, int n);        // adds
    static double noteHz(int group);

private:
    double sr_ = 48000;
    double phase_[kDrums], inc_[kDrums];
    double target_[kDrums], amp_[kDrums];
    double smooth_ = 0, enable_ = 0, enableTarget_ = 0, level_ = 0.25;
};

// ------------------------------------------------------------------------------------------------ Sidechain
// Fly Sidechain (#21): a motor group's standardised rate drives a gain on the plugin's output. Duck: gain =
// 1 - amount x z', gate: gain = amount x z', with z' = tanh(max(z, 0) / 2) in [0, 1). 10 ms attack (gain
// falling), 80 ms release (gain rising). The gain always lies in [0, 1].
class Sidechain {
public:
    enum Mode { Duck = 0, Gate = 1 };
    Sidechain();
    void prepare(double sampleRate, int maxBlock);
    void reset();
    void setAmount(float amount01);
    void setMode(int mode);
    void setGroupRate(float z);                     // once per 5 ms frame
    void process(float* L, float* R, int n);        // multiplies
    float currentGain() const { return float(gain_); }

private:
    double sr_ = 48000, attack_ = 0, release_ = 0, gain_ = 1, target_ = 1, amount_ = 0, z_ = 0;
    int mode_ = Duck;
};

// ------------------------------------------------------------------------------------------------ Limiter
// Brick wall at -0.3 dBFS: a gain envelope with 1 ms attack and 50 ms release follows min(1, ceiling / peak),
// and a hard clip at the ceiling guarantees that no output sample ever exceeds it (no look-ahead, so the clip
// is what makes the guarantee; the envelope keeps it from being audible on ordinary peaks).
class Limiter {
public:
    Limiter();
    void prepare(double sampleRate, int maxBlock);
    void reset();
    void process(float* L, float* R, int n);
    float ceiling() const { return float(ceiling_); }
    float currentGain() const { return float(gain_); }

private:
    double sr_ = 48000, ceiling_ = 0.966051, gain_ = 1, attack_ = 0, release_ = 0;
};

}  // namespace voices
}  // namespace flybrain
