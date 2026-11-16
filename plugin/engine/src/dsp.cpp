#include "flybrain/dsp.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace flybrain {

// ---------------------------------------------------------------- resampler
namespace {
double besselI0(double x) {
    double sum = 1.0, term = 1.0;
    const double y = x * x / 4.0;
    for (int k = 1; k < 200; ++k) {
        term *= y / (double(k) * double(k));
        sum += term;
        if (term < 1e-17 * sum) break;
    }
    return sum;
}
}  // namespace

void Resampler::prepare(int hostRate, int outRate) {
    if (hostRate <= 0 || outRate <= 0) throw std::runtime_error("resampler: bad rate");
    int g = std::gcd(hostRate, outRate);
    L = outRate / g;
    M = hostRate / g;
    passThrough = (L == 1 && M == 1);
    proto.clear();
    hist.clear();
    if (!passThrough) {
        // prototype at the upsampled rate Fu = hostRate * L; cutoff centred so the stopband starts at outRate/2
        const double fu = double(hostRate) * L;
        const double fc = outRate / 2.0 - kTransitionHz / 2.0;      // Hz
        const double cut = 2.0 * fc / fu;                             // cycles per upsampled sample, x2
        const int half = kHalfTaps * L;
        proto.resize(2 * half + 1);
        const double i0b = besselI0(kBeta);
        for (int i = -half; i <= half; ++i) {
            double x = double(i);
            double s = (i == 0) ? cut : std::sin(M_PI * cut * x) / (M_PI * x);
            double r = x / double(half);
            double w = besselI0(kBeta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0b;
            proto[i + half] = double(L) * s * w;
        }
        histLen = 2 * kHalfTaps + 2;
        hist.assign(histLen, 0.0f);
    }
    reset();
}

void Resampler::reset() {
    for (auto& h : hist) h = 0.0f;
    nIn = 0; nextK = 0; nextNum = 0; ringPos = 0;
}

int Resampler::maxOutputs(int nInputs) const {
    if (passThrough) return nInputs;
    return int((int64_t(nInputs) * L) / M) + 2;
}

double Resampler::computeOutput(int64_t k) const {
    // y[k] = sum_j x[q - j] h[p + j L],  q = floor(kM / L), p = kM mod L; j from -H..H as long as the tap exists
    const int64_t num = k * M;
    const int64_t q = num / L;
    const int p = int(num - q * L);
    const int half = kHalfTaps * L;
    double acc = 0.0;
    for (int j = -kHalfTaps; j <= kHalfTaps; ++j) {
        int tap = p + j * L;
        if (tap < -half || tap > half) continue;
        int64_t idx = q - j;                            // input sample index
        if (idx < 0) continue;                           // before the stream: zeros
        // hist holds inputs (nIn - histLen, nIn]; idx is within it by construction
        int slot = int((idx % histLen + histLen) % histLen);
        acc += double(hist[slot]) * proto[tap + half];
    }
    return acc;
}

int Resampler::process(const float* in, int n, float* out) {
    if (passThrough) {
        for (int i = 0; i < n; ++i) out[i] = in[i];
        return n;
    }
    int produced = 0;
    for (int i = 0; i < n; ++i) {
        hist[size_t(nIn % histLen)] = in[i];
        ++nIn;
        // every output k whose last needed input q + H has arrived (q + H <= nIn - 1)
        for (;;) {
            int64_t q = nextNum / L;
            if (q + kHalfTaps > nIn - 1) break;
            out[produced++] = float(computeOutput(nextK));
            ++nextK;
            nextNum += M;
        }
    }
    return produced;
}

// ---------------------------------------------------------------- FFT
void RealFFT::prepare(int size) {
    n = size;
    re.assign(n, 0); im.assign(n, 0);
    cosT.resize(n / 2); sinT.resize(n / 2);
    for (int k = 0; k < n / 2; ++k) { cosT[k] = std::cos(-2 * M_PI * k / n); sinT[k] = std::sin(-2 * M_PI * k / n); }
    rev.resize(n);
    int bits = 0;
    while ((1 << bits) < n) ++bits;
    for (int i = 0; i < n; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        rev[i] = r;
    }
}

void RealFFT::forward(const double* in, double* outRe, double* outIm) {
    for (int i = 0; i < n; ++i) { re[rev[i]] = in[i]; im[rev[i]] = 0.0; }
    for (int len = 2; len <= n; len <<= 1) {
        int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                double wr = cosT[j * step], wi = sinT[j * step];
                double xr = re[i + j + len / 2], xi = im[i + j + len / 2];
                double tr = xr * wr - xi * wi, ti = xr * wi + xi * wr;
                re[i + j + len / 2] = re[i + j] - tr; im[i + j + len / 2] = im[i + j] - ti;
                re[i + j] += tr; im[i + j] += ti;
            }
        }
    }
    for (int k = 0; k <= n / 2; ++k) { outRe[k] = re[k]; outIm[k] = im[k]; }
}

// ---------------------------------------------------------------- ear
void Ear::prepare(const BrainFile& b) {
    nfft = b.nFft; hop = b.hop; nb = b.nBands;
    window.resize(nfft);
    for (int k = 0; k < nfft; ++k) window[k] = 0.5 - 0.5 * std::cos(2 * M_PI * k / nfft);   // torch.hann_window (periodic)
    frame.assign(nfft, 0.0);
    scratch.assign(nfft, 0.0);
    re.assign(nfft / 2 + 1, 0.0); im.assign(nfft / 2 + 1, 0.0); power.assign(nfft / 2 + 1, 0.0);
    binsOfBand.assign(nb, {});
    for (int bin = 0; bin <= nfft / 2; ++bin)
        for (int band = 0; band < nb; ++band)
            if (b.bandMatrix[size_t(bin) * nb + band] != 0.0f) binsOfBand[band].push_back(bin);
    fft.prepare(nfft);
    reset();
    filled = 0;
}

void Ear::reset() {
    for (int k = 0; k < nfft - hop; ++k) frame[k] = 0.0;     // the history; samples pending toward the next hop stay
}

void Ear::clearPending() { filled = 0; }

bool Ear::push(float sample, double* bands) {
    // the frame holds the last nfft samples; new samples enter at the end
    // (shift once per hop instead of per sample: keep `filled` new samples at the tail)
    frame[size_t(nfft - hop + filled)] = double(sample);
    ++filled;
    if (filled < hop) return false;
    filled = 0;
    std::vector<double>& f = frame;
    // windowed copy into a scratch (reuse `power` as scratch is too small: use re as temp of size nfft? keep a loop)
    for (int k = 0; k < nfft; ++k) scratch[k] = f[k] * window[k];
    fft.forward(scratch.data(), re.data(), im.data());
    for (int k = 0; k <= nfft / 2; ++k) power[k] = re[k] * re[k] + im[k] * im[k];
    for (int band = 0; band < nb; ++band) {
        double s = 0.0;
        for (int bin : binsOfBand[band]) s += power[bin];
        bands[band] = std::log1p(s);
    }
    // slide: the last nfft - hop samples move to the front
    for (int k = 0; k < nfft - hop; ++k) f[k] = f[k + hop];
    return true;
}

}  // namespace flybrain
