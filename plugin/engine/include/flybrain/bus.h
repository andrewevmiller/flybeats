// Wait-free exchange between the audio thread and the others: a triple buffer (used for the snapshot tap and
// for the control block) and a single-producer single-consumer ring (the recorder tap).
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>

namespace flybrain {

// Three slots. The writer owns one, the reader owns one, the third is in the middle with a "fresh" flag.
// publish() and acquire() are one atomic exchange each; neither side ever waits for the other.
template <class Slot>
class TripleBuffer {
public:
    Slot& writeSlot() { return slots[writeIdx]; }
    void publish() {
        int old = middle.exchange(writeIdx | kFresh, std::memory_order_acq_rel);
        writeIdx = old & 3;
    }
    // Returns true if a newer slot was taken; readSlot() then holds it.
    bool acquire() {
        if (!(middle.load(std::memory_order_acquire) & kFresh)) return false;
        int old = middle.exchange(readIdx, std::memory_order_acq_rel);
        readIdx = old & 3;
        return true;
    }
    Slot& readSlot() { return slots[readIdx]; }
    Slot slots[3];

private:
    static constexpr int kFresh = 4;
    int writeIdx = 0, readIdx = 2;
    std::atomic<int> middle{1};
};

// Fixed-capacity SPSC ring of T. push() never blocks: it drops when full and counts the drop.
template <class T>
class SpscRing {
public:
    void allocate(size_t capacity) { buf.resize(capacity); head.store(0); tail.store(0); dropped.store(0); }
    size_t capacity() const { return buf.size(); }
    // producer: get the slot to fill, or nullptr when full
    T* beginPush() {
        size_t t = tail.load(std::memory_order_relaxed), h = head.load(std::memory_order_acquire);
        if (t - h >= buf.size()) { dropped.fetch_add(1, std::memory_order_relaxed); return nullptr; }
        return &buf[t % buf.size()];
    }
    void endPush() { tail.store(tail.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
    // consumer
    T* front() {
        size_t h = head.load(std::memory_order_relaxed), t = tail.load(std::memory_order_acquire);
        if (h == t) return nullptr;
        return &buf[h % buf.size()];
    }
    void pop() { head.store(head.load(std::memory_order_relaxed) + 1, std::memory_order_release); }
    size_t size() const { return tail.load(std::memory_order_acquire) - head.load(std::memory_order_acquire); }
    uint64_t droppedCount() const { return dropped.load(std::memory_order_relaxed); }
    std::vector<T>& storage() { return buf; }

private:
    std::vector<T> buf;
    std::atomic<size_t> head{0}, tail{0};
    std::atomic<uint64_t> dropped{0};
};

// One frame as the UI and the recorder see it. Values in double whatever the engine precision (exact for float).
struct Snapshot {
    int64_t frame = -1;
    bool resetFlag = false, runaway = false, neutral = true, muted = false;
    uint64_t controlsSerial = 0;
    std::vector<double> bands, ear, rates, mn, prob, vel;
    std::vector<uint8_t> hits;
    void allocate(int nBands, int n, int nMn, int nDrums) {
        bands.assign(size_t(nBands), 0); ear.assign(size_t(nBands), 0); rates.assign(size_t(n), 0);
        mn.assign(size_t(nMn), 0); prob.assign(size_t(nDrums), 0); vel.assign(size_t(nDrums), 0); hits.assign(size_t(nDrums), 0);
    }
    void copyFrom(const Snapshot& o) {   // sizes equal: no allocation
        frame = o.frame; resetFlag = o.resetFlag; runaway = o.runaway; neutral = o.neutral; muted = o.muted; controlsSerial = o.controlsSerial;
        bands = o.bands; ear = o.ear; rates = o.rates; mn = o.mn; prob = o.prob; vel = o.vel; hits = o.hits;
    }
};

}  // namespace flybrain
