// Streaming DSP: the rational polyphase resampler, a radix-2 real FFT and the 16-band ear.
#pragma once
#include <cstdint>
#include <vector>

#include "flybrain/brainfile.h"

namespace flybrain {

// Host rate -> 16 kHz. Rational ratio L/M from the integer rates; windowed-sinc (Kaiser) prototype at the
// upsampled rate, evaluated per phase. Output sample k uses input samples q-H .. q+H around q = floor(k*M/L),
// so it is produced when input q+H has arrived: a fixed latency of H input samples (plus the phase fraction).
// The same arithmetic whatever the block size, so the output is bit-identical. plugin/export/flybrain/resample.py
// is the numpy port of exactly this algorithm.
class Resampler {
public:
    static constexpr int kHalfTaps = 96;    // H: taps per side per output sample
    static constexpr double kBeta = 8.96;   // Kaiser beta: about 90 dB stopband
    static constexpr double kTransitionHz = 1300.0;   // the stopband starts at 8 kHz (the 16 kHz Nyquist)

    Resampler() = default;
    void prepare(int hostRate, int outRate = 16000);
    void reset();
    // Feeds `n` input samples; returns the number of output samples written to `out` (capacity must be at least
    // maxOutputs(n)). No allocation.
    int process(const float* in, int n, float* out);
    int maxOutputs(int nInputs) const;      // an upper bound, for buffer sizing
    int latencyInputSamples() const { return passThrough ? 0 : kHalfTaps; }
    bool isPassThrough() const { return passThrough; }
    int upFactor() const { return L; }
    int downFactor() const { return M; }
    // the prototype (length (2H+1)*L, centred) and phase layout, for the numpy port's test
    const std::vector<double>& prototype() const { return proto; }

private:
    bool passThrough = true;
    int L = 1, M = 1;
    std::vector<double> proto;          // h[i] for i in [-(H*L), H*L], index i + H*L
    std::vector<float> hist;            // ring of the last 2H+2 input samples
    int histLen = 0;
    int64_t nIn = 0;                    // input samples consumed so far
    int64_t nextK = 0;                  // next output index
    int64_t nextNum = 0;                // nextK * M (kept incrementally)
    int ringPos = 0;
    double computeOutput(int64_t k) const;
};

// Real FFT of size N = 2^p through a complex radix-2 transform, in double.
class RealFFT {
public:
    void prepare(int n);
    // in: n real samples; outRe/outIm: n/2+1 bins
    void forward(const double* in, double* outRe, double* outIm);
    int size() const { return n; }

private:
    int n = 0;
    std::vector<double> re, im, cosT, sinT;
    std::vector<int> rev;
};

// Mono 16 kHz samples -> one feature vector per hop, as reference.py's Ear.
class Ear {
public:
    void prepare(const BrainFile& b);
    void reset();           // zero history (what a reset does); pending samples are kept
    void clearPending();
    // Feeds one 16 kHz sample; returns true when a frame completed, with the raw features log1p(band power) in
    // `bands` (double); normalisation and the learned gain/offset are applied by the network in its precision.
    bool push(float sample, double* bands);
    int nBands() const { return nb; }

private:
    int nfft = 0, hop = 0, nb = 0;
    std::vector<double> window, frame, scratch, re, im, power;
    std::vector<std::vector<int>> binsOfBand;
    int filled = 0;         // samples accumulated toward the next hop
    RealFFT fft;
};

}  // namespace flybrain
