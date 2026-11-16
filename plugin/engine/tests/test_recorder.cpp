// The recorder: 60 s recorded through the ring equals the frames seen through the snapshot, 0 dropped.
#include <cstring>

#include "data.h"

int main(int, char** argv) {
    std::string dir = argv[1];
    if (!exists(dir + "/small.flybrain")) { std::printf("SKIP: no test data\n"); return 0; }
    BrainFile b = BrainFile::load(dir + "/small.flybrain");
    std::mt19937 rng(9); std::normal_distribution<float> nd(0.0f, 0.1f);
    std::vector<float> audio(16000 * 60); for (auto& v : audio) v = nd(rng);
    Engine e(b); EngineConfig cfg; cfg.hostRate = 16000; cfg.maxBlock = 512; cfg.doublePrecision = true; cfg.prerollFrames = 0; cfg.recorderCapacityFrames = 1024; e.prepare(cfg);
    std::string bin = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/flybrain_rec_test.bin";
    Json info; info.type = Json::Object; Json s; s.type = Json::String; s.str = b.sha256; info.obj["brain_sha256"] = s;
    e.recorder().start(bin, bin + ".json", info);
    Controls c(b); c.gainScale = 1.1;
    Snapshot snap; e.allocateSnapshot(snap); HitEvent ev[256];
    std::vector<double> rates; std::vector<uint8_t> hits; std::vector<int64_t> frames;
    for (size_t p = 0; p + 80 <= audio.size(); p += 80) {
        const float* ch[1] = {audio.data() + p};
        e.process(ch, 1, 80, ev, 256);
        while (e.readSnapshot(snap)) { frames.push_back(snap.frame); rates.insert(rates.end(), snap.rates.begin(), snap.rates.end()); hits.insert(hits.end(), snap.hits.begin(), snap.hits.end()); }
        if (p == 16000 * 30) e.setControls(c, "gain 1.1");
    }
    e.flush(ev, 256);
    while (e.readSnapshot(snap)) { frames.push_back(snap.frame); rates.insert(rates.end(), snap.rates.begin(), snap.rates.end()); hits.insert(hits.end(), snap.hits.begin(), snap.hits.end()); }
    e.recorder().stop();
    CHECK_MSG(e.recorder().framesDropped() == 0, "dropped %llu", (unsigned long long)e.recorder().framesDropped());
    CHECK_MSG(e.recorder().framesWritten() == frames.size(), "written %llu, seen %zu", (unsigned long long)e.recorder().framesWritten(), frames.size());
    // parse the file
    std::vector<uint8_t> raw = readFile(bin);
    CHECK(std::memcmp(raw.data(), "FLYREC01", 8) == 0);
    size_t rec = 8 + 4 + 8 + 8 * (16 + 16 + size_t(b.n) + size_t(b.nMn()) + 2 * size_t(b.nDrums)) + size_t(b.nDrums);
    size_t count = (raw.size() - 24) / rec;
    CHECK(count == frames.size());
    bool same = true;
    for (size_t f = 0; f < count && same; ++f) {
        const uint8_t* p = raw.data() + 24 + f * rec;
        int64_t fr; std::memcpy(&fr, p, 8);
        same = fr == frames[f];
        const uint8_t* rp = p + 8 + 4 + 8 + 8 * 32;
        same = same && std::memcmp(rp, rates.data() + f * size_t(b.n), 8 * size_t(b.n)) == 0;
        const uint8_t* hp = p + rec - size_t(b.nDrums);
        same = same && std::memcmp(hp, hits.data() + f * size_t(b.nDrums), size_t(b.nDrums)) == 0;
    }
    CHECK(same);
    auto side = readFile(bin + ".json");
    Json j = Json::parse(std::string(side.begin(), side.end()));
    CHECK(j["frames_written"].asInt() == int64_t(frames.size()) && j["frames_dropped"].asInt() == 0);
    CHECK(j["control_log"].size() == 1 && j["control_log"].at(0)["controls"].asString() == "gain 1.1");
    CHECK(j["brain_sha256"].asString() == b.sha256);
    std::printf("recorder: %zu frames (60 s) written and equal to the snapshots, 0 dropped, 1 control change logged at frame %lld\n", count, (long long)j["control_log"].at(0)["frame"].asInt());
    // a tiny ring under a stalled consumer drops and counts, and never blocks the audio thread
    {
        Engine e2(b); EngineConfig c2 = cfg; c2.recorderCapacityFrames = 4; e2.prepare(c2);
        e2.recorder().start(bin, bin + ".json", info);
        for (size_t p = 0; p + 80 <= 16000; p += 80) { const float* ch[1] = {audio.data() + p}; e2.process(ch, 1, 80, ev, 256); }
        e2.recorder().stop();
        std::printf("4-frame ring over 1 s: written %llu, dropped %llu\n", (unsigned long long)e2.recorder().framesWritten(), (unsigned long long)e2.recorder().framesDropped());
        CHECK(e2.recorder().framesWritten() + e2.recorder().framesDropped() == 199);
    }
    return finish();
}
