// The network in one precision T: sparse core, readout with history, peak decoder, runaway guard.
// Every operation is in the order reference.py uses, so the two agree bit for bit in the same precision.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "flybrain/brainfile.h"
#include "flybrain/controls.h"

namespace flybrain {

template <class T>
inline T sigmoidT(T x) {
    if (x >= T(0)) return T(1) / (T(1) + std::exp(-x));
    T e = std::exp(x);
    return e / (T(1) + e);
}

// The control block compiled for the audio thread: weights and masks ready to use. Built off the audio thread.
template <class T>
struct CompiledControls {
    std::vector<T> wValues;        // nnz, after gain, sign flips and un-silencing
    std::vector<T> alive;          // n
    std::vector<T> alpha;          // n
    std::vector<T> earMute;        // nEars
    std::vector<T> bandOffset;     // nBands
    std::vector<Injection> inject; // copied to the audio side at swap
    T threshold = T(0.5);
    bool neutral = true;
    uint64_t serial = 0;

    void allocate(const BrainFile& b) {
        wValues.assign(b.wValues.begin(), b.wValues.end());
        alive.assign(b.alive.begin(), b.alive.end());
        alpha.assign(b.alpha.begin(), b.alpha.end());
        earMute.assign(size_t(b.nEars()), T(1));
        bandOffset.assign(size_t(b.nBands), T(0));
        inject.reserve(kMaxInjections);
        threshold = T(b.hitThreshold);
    }

    // reference.py Core.configure, in T
    void compile(const BrainFile& b, const Controls& c) {
        const size_t nnz = b.wValues.size(), n = size_t(b.n);
        bool anyUnsilence = false, anyFlip = false;
        for (size_t i = 0; i < n; ++i) { anyUnsilence |= c.unsilence[i] != 0; anyFlip |= c.signFlip[i] != 1.0f; }
        for (size_t k = 0; k < nnz; ++k) wValues[k] = T(b.wValues[k]);
        if (anyUnsilence) {
            for (size_t k = 0; k < nnz; ++k) {
                int j = b.wIndices[k];
                if (c.unsilence[j] != 0 && b.sign[j] == 0) wValues[k] = T(int(c.unsilence[j])) * T(b.wUnsigned[k]);
            }
        }
        if (c.gainScale != 1.0 || anyFlip) {
            const T g = T(c.gainScale);
            for (size_t k = 0; k < nnz; ++k) {
                T preMult = T(c.signFlip[b.wIndices[k]]) * g;
                wValues[k] = wValues[k] * preMult;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            T a = T(b.alive[i]);
            if (c.unsilence[i] != 0 && b.sign[i] == 0) a = T(1);
            alive[i] = a * T(c.lesion[i]);
        }
        if (c.tauScale == 1.0) {
            for (size_t i = 0; i < n; ++i) alpha[i] = T(b.alpha[i]);
        } else {
            for (size_t i = 0; i < n; ++i) {
                double tau = double(b.tauMs[i]) * c.tauScale;
                tau = std::min(std::max(tau, b.tauMin), b.tauMax);
                alpha[i] = T(float(b.frameMs / tau));
            }
        }
        for (size_t e = 0; e < earMute.size(); ++e) earMute[e] = T(c.earMute[e]);
        for (size_t k = 0; k < bandOffset.size(); ++k) bandOffset[k] = T(c.bandOffset[k]);
        inject.clear();
        for (size_t i = 0; i < c.nInject && i < kMaxInjections; ++i) inject.push_back(c.inject[i]);
        threshold = T(c.threshold);
        neutral = c.isNeutral(b);
    }
};

template <class T>
class Core {
public:
    void prepare(const BrainFile& b) {
        n = b.n;
        indptr = &b.wIndptr; indices = &b.wIndices;
        bias.assign(b.bias.begin(), b.bias.end());
        earIdx = &b.earIdx;
        v.assign(size_t(n), T(0)); r.assign(size_t(n), T(0)); drive.assign(size_t(n), T(0)); rec.assign(size_t(n), T(0));
        active.reserve(kMaxInjections);
        reset();
    }
    void reset() {
        std::fill(v.begin(), v.end(), T(0));
        std::fill(r.begin(), r.end(), T(0));
        active.clear();
    }
    void setControls(const CompiledControls<T>* c) {
        ctl = c;
        // injections with a start frame in the future wait; the rest start now
        active.clear();
        for (const Injection& inj : c->inject) if (inj.frames > 0) active.push_back(inj);
    }
    // earIn: current per ear neuron (nEars), already masked by the caller? No: masked here with ctl->earMute.
    void step(const T* earIn, int64_t frameIndex) {
        std::fill(drive.begin(), drive.end(), T(0));
        const auto& ei = *earIdx;
        for (size_t e = 0; e < ei.size(); ++e) drive[size_t(ei[e])] = earIn[e] * ctl->earMute[e];
        for (size_t i = 0; i < active.size();) {
            Injection& inj = active[i];
            if (inj.startFrame >= 0 && frameIndex < inj.startFrame) { ++i; continue; }
            drive[size_t(inj.neuron)] += T(inj.current);
            if (--inj.frames <= 0) active.erase(active.begin() + long(i));   // bounded: at most kMaxInjections
            else ++i;
        }
        const int32_t* ip = indptr->data();
        const int32_t* ix = indices->data();
        const T* w = ctl->wValues.data();
        for (int i = 0; i < n; ++i) {
            T sum = T(0);
            for (int32_t k = ip[i]; k < ip[i + 1]; ++k) sum += w[k] * r[size_t(ix[k])];
            rec[size_t(i)] = sum;
        }
        const T* al = ctl->alpha.data();
        const T* alive = ctl->alive.data();
        for (size_t i = 0; i < size_t(n); ++i) {
            T t = -v[i] + rec[i];
            t = t + bias[i];
            t = t + drive[i];
            t = t * al[i];
            v[i] = v[i] + t;
            r[i] = std::max(v[i], T(0)) * alive[i];
        }
    }
    std::vector<T> v, r;
    int size() const { return n; }

private:
    int n = 0;
    const std::vector<int32_t>* indptr = nullptr;
    const std::vector<int32_t>* indices = nullptr;
    const std::vector<int32_t>* earIdx = nullptr;
    std::vector<T> bias, drive, rec;
    std::vector<Injection> active;
    const CompiledControls<T>* ctl = nullptr;
};

template <class T>
class Readout {
public:
    void prepare(const BrainFile& b) {
        mnIdx = &b.mnIdx; mnDrum = &b.mnDrum;
        nd = b.nDrums; nm = b.nMn();
        mean.assign(b.mnMean.begin(), b.mnMean.end()); stdv.assign(b.mnStd.begin(), b.mnStd.end());
        wHit.assign(b.wHit.begin(), b.wHit.end()); uVel.assign(b.uVel.begin(), b.uVel.end());
        cHit.assign(b.cHit.begin(), b.cHit.end()); eVel.assign(b.eVel.begin(), b.eVel.end());
        blocks.assign(b.historyBlocks.begin(), b.historyBlocks.end());
        edges.assign(blocks.size() + 1, 0);
        for (size_t j = 0; j < blocks.size(); ++j) edges[j + 1] = edges[j] + blocks[j];
        back = edges.back();
        const size_t nf = 1 + blocks.size();
        // per readout neuron: that neuron's drum's profile row
        pHit.assign(size_t(nm) * nf, T(0)); pVel.assign(size_t(nm) * nf, T(0));
        for (int i = 0; i < nm; ++i)
            for (size_t k = 0; k < nf; ++k) {
                pHit[size_t(i) * nf + k] = T(b.pHit[size_t((*mnDrum)[i]) * nf + k]);
                pVel[size_t(i) * nf + k] = T(b.pVel[size_t((*mnDrum)[i]) * nf + k]);
            }
        hist.assign(size_t(std::max(back, 1)) * size_t(nm), T(0));
        mn.assign(size_t(nm), T(0)); z.assign(size_t(nm), T(0)); effHit.assign(size_t(nm), T(0)); effVel.assign(size_t(nm), T(0));
        blockMean.assign(size_t(nm), T(0));
        logits.assign(size_t(nd), T(0)); prob.assign(size_t(nd), T(0)); vel.assign(size_t(nd), T(0)); vlog.assign(size_t(nd), T(0));
        reset();
    }
    void reset() {
        std::fill(hist.begin(), hist.end(), T(0));
        pos = 0;
    }
    // rates: all neurons' rates after the guard's clamp
    void step(const T* rates) {
        const size_t nf = 1 + blocks.size();
        for (int i = 0; i < nm; ++i) { mn[i] = rates[size_t((*mnIdx)[i])]; z[i] = (mn[i] - mean[i]) / stdv[i]; }
        // feats[0] = z: eff = 0 + z * p[:,0]  (python's sum starts at int 0: exact)
        for (int i = 0; i < nm; ++i) { effHit[i] = z[i] * pHit[size_t(i) * nf]; effVel[i] = z[i] * pVel[size_t(i) * nf]; }
        for (size_t j = 0; j < blocks.size(); ++j) {
            // block j: lags edges[j]+1 .. edges[j+1]; sum rows in that order (numpy's axis-0 sum is sequential)
            std::fill(blockMean.begin(), blockMean.end(), T(0));
            bool first = true;
            for (int lag = edges[j] + 1; lag <= edges[j + 1]; ++lag) {
                int idx = ((pos - lag) % back + back) % back;
                const T* row = &hist[size_t(idx) * size_t(nm)];
                if (first) { for (int i = 0; i < nm; ++i) blockMean[i] = row[i]; first = false; }
                else for (int i = 0; i < nm; ++i) blockMean[i] = blockMean[i] + row[i];
            }
            const T nj = T(blocks[j]);
            for (int i = 0; i < nm; ++i) {
                T m = blockMean[i] / nj;
                effHit[i] = effHit[i] + m * pHit[size_t(i) * nf + j + 1];
                effVel[i] = effVel[i] + m * pVel[size_t(i) * nf + j + 1];
            }
        }
        if (back > 0) {
            T* row = &hist[size_t(pos) * size_t(nm)];
            for (int i = 0; i < nm; ++i) row[i] = z[i];
            pos = (pos + 1) % back;
        }
        // (eff * w) @ onehot: per drum, the sum over its neurons in index order
        for (int d = 0; d < nd; ++d) { logits[d] = T(0); vlog[d] = T(0); }
        for (int i = 0; i < nm; ++i) {
            int d = (*mnDrum)[i];
            logits[d] += effHit[i] * wHit[i];
            vlog[d] += effVel[i] * uVel[i];
        }
        for (int d = 0; d < nd; ++d) {
            prob[d] = sigmoidT(logits[d] + cHit[d]);
            vel[d] = sigmoidT(vlog[d] + eVel[d]);
        }
    }
    std::vector<T> mn, z, prob, vel;

private:
    const std::vector<int32_t>* mnIdx = nullptr;
    const std::vector<int32_t>* mnDrum = nullptr;
    int nd = 0, nm = 0, back = 0, pos = 0;
    std::vector<T> mean, stdv, wHit, uVel, cHit, eVel, pHit, pVel, hist, effHit, effVel, blockMean, logits, vlog;
    std::vector<int> blocks, edges;
};

// loss.peaks, streaming: frame t is a hit if p[t] >= thr, p[t] >= p[t-1], p[t] > p[t+1]; decided when t+1 arrives.
template <class T>
class PeakDecoder {
public:
    void prepare(int nDrums) {
        nd = nDrums;
        prev.assign(size_t(nd), T(0)); cur.assign(size_t(nd), T(0)); curVel.assign(size_t(nd), T(0));
        hits.assign(size_t(nd), 0); decidedVel.assign(size_t(nd), T(0));
        reset();
    }
    void reset() {
        std::fill(prev.begin(), prev.end(), -std::numeric_limits<T>::infinity());
        haveCur = false;
        curFrame = -1;
    }
    // Returns true if a frame was decided: its index in decidedFrame, hits[] and the velocities in decidedVel.
    bool push(const T* p, const T* v, int64_t frame, T threshold) {
        bool decided = false;
        if (haveCur) {
            for (int d = 0; d < nd; ++d) hits[d] = (cur[d] >= threshold && cur[d] >= prev[d] && cur[d] > p[d]) ? 1 : 0;
            decidedFrame = curFrame;
            decided = true;
            std::swap(prev, cur);         // prev = cur (values), cur reused below
            std::swap(decidedVel, curVel);
        }
        for (int d = 0; d < nd; ++d) { cur[d] = p[d]; curVel[d] = v[d]; }
        curFrame = frame;
        haveCur = true;
        return decided;
    }
    bool flush(T threshold) {
        if (!haveCur) return false;
        for (int d = 0; d < nd; ++d) hits[d] = (cur[d] >= threshold && cur[d] >= prev[d]) ? 1 : 0;
        decidedFrame = curFrame;
        std::swap(decidedVel, curVel);
        reset();
        return true;
    }
    // Decide the pending frame with no hits (used when a runaway reset interrupts the stream).
    bool dropPending(int64_t& frame) {
        if (!haveCur) return false;
        for (int d = 0; d < nd; ++d) hits[d] = 0;
        frame = curFrame;
        reset();
        return true;
    }
    std::vector<uint8_t> hits;
    std::vector<T> decidedVel;
    int64_t decidedFrame = -1;

private:
    int nd = 0;
    std::vector<T> prev, cur, curVel;
    bool haveCur = false;
    int64_t curFrame = -1;
};

template <class T>
class RunawayGuard {
public:
    void prepare(const BrainFile& b) {
        enabled = b.runaway.set;
        ceiling = enabled ? T(b.runaway.rateCeiling) : std::numeric_limits<T>::infinity();
        meanLimit = enabled ? b.runaway.meanRateLimit : std::numeric_limits<double>::infinity();
        frames = b.runaway.frames;
        reset();
    }
    void setLimits(double rateCeiling, double meanRateLimit, int nFrames) {
        enabled = true; ceiling = T(rateCeiling); meanLimit = meanRateLimit; frames = nFrames;
    }
    void reset() { over = 0; }
    // Clamps v and r; returns true when the runaway rule triggers (the caller resets).
    bool check(std::vector<T>& v, std::vector<T>& r) {
        bool finite = true;
        double sum = 0.0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i] > ceiling) v[i] = ceiling;
            if (r[i] > ceiling) r[i] = ceiling;
            finite = finite && std::isfinite(double(v[i]));
            sum += double(r[i]);
        }
        bool bad = !finite || (sum / double(v.size())) > meanLimit;
        over = bad ? over + 1 : 0;
        if (over >= frames) { over = 0; ++triggered; return true; }
        return false;
    }
    int triggered = 0;

private:
    bool enabled = false;
    T ceiling = std::numeric_limits<T>::infinity();
    double meanLimit = std::numeric_limits<double>::infinity();
    int frames = 20, over = 0;
};

}  // namespace flybrain
