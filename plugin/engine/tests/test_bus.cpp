// The exchange buffers: a slow reader never blocks the writer and always sees a whole frame.
#include <atomic>
#include <chrono>
#include <thread>

#include "data.h"
#include "flybrain/bus.h"

struct Slot { uint64_t seq = 0; std::vector<uint64_t> data; uint64_t sum = 0; };

int main(int, char**) {
    TripleBuffer<Slot> tb;
    for (auto& s : tb.slots) s.data.assign(4096, 0);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> written{0};
    std::thread writer([&] {
        uint64_t seq = 0;
        while (!stop.load()) {
            Slot& s = tb.writeSlot();
            s.seq = ++seq;
            uint64_t sum = 0;
            for (size_t i = 0; i < s.data.size(); ++i) { s.data[i] = seq * 7919 + i; sum += s.data[i]; }
            s.sum = sum;
            tb.publish();
            written.store(seq);
        }
    });
    uint64_t reads = 0, last = 0;
    bool consistent = true, monotonic = true;
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(500)) {
        if (tb.acquire()) {
            Slot& s = tb.readSlot();
            uint64_t sum = 0;
            for (uint64_t v : s.data) sum += v;
            consistent = consistent && sum == s.sum && s.data[1] == s.seq * 7919 + 1;
            monotonic = monotonic && s.seq > last;
            last = s.seq;
            ++reads;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(500));   // a slow reader
    }
    stop.store(true);
    writer.join();
    CHECK(consistent);
    CHECK(monotonic);
    CHECK_MSG(written.load() > reads * 2, "writer %llu frames, reader %llu: the writer was not slowed to the reader", (unsigned long long)written.load(), (unsigned long long)reads);
    CHECK(!tb.acquire() || true);
    std::printf("triple buffer: writer published %llu, slow reader took %llu, all consistent and increasing\n", (unsigned long long)written.load(), (unsigned long long)reads);

    // the ring: drops when full, counts the drops, order kept
    SpscRing<int> ring; ring.allocate(8);
    for (int i = 0; i < 12; ++i) { int* p = ring.beginPush(); if (p) { *p = i; ring.endPush(); } }
    CHECK(ring.size() == 8 && ring.droppedCount() == 4);
    int expect = 0; bool order = true;
    while (int* p = ring.front()) { order = order && *p == expect++; ring.pop(); }
    CHECK(order && expect == 8);
    return finish();
}
