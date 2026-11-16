// flybrain_render: a brain and an audio file in, decoded hits (JSON) out, optional per-frame dump and recording.
//   flybrain_render brain.flybrain --audio in.f32 --rate 48000 [--block 512] [--double] [--no-ftz]
//                   [--preroll N] [--trim dB] [--controls c.json] [--dump frames.bin] [--record rec.bin]
//                   [--features ear.f64 | --features32 ear.f32]  (bypass resampler and ear)
// Input: raw little-endian float32 mono (.f32), or a 16-bit / 32-bit-float PCM WAV (channels averaged).
// Dump: "FLYDUMP1", uint32 n_bands,n,n_mn,n_drums, then per frame: int64 frame, uint8 reset, runaway, neutral,
// muted, float64 bands, ear, rates, mn, prob, vel, uint8 hits (the recorder's record without the serial).
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "flybrain/engine.h"
#include "flybrain/util.h"

using namespace flybrain;

static std::vector<float> loadAudio(const std::string& path, int& rate, int givenRate) {
    std::vector<uint8_t> raw = readFile(path);
    std::vector<float> out;
    if (raw.size() >= 44 && std::memcmp(raw.data(), "RIFF", 4) == 0 && std::memcmp(raw.data() + 8, "WAVE", 4) == 0) {
        size_t p = 12;
        int channels = 1, bits = 16, format = 1;
        while (p + 8 <= raw.size()) {
            std::string id(reinterpret_cast<const char*>(raw.data() + p), 4);
            uint32_t len = readU32(raw.data() + p + 4);
            p += 8;
            if (id == "fmt ") {
                format = raw[p] | raw[p + 1] << 8;
                channels = raw[p + 2] | raw[p + 3] << 8;
                rate = int(readU32(raw.data() + p + 4));
                bits = raw[p + 14] | raw[p + 15] << 8;
            } else if (id == "data") {
                size_t frames = len / size_t(channels * bits / 8);
                out.resize(frames);
                for (size_t i = 0; i < frames; ++i) {
                    double sum = 0;
                    for (int c = 0; c < channels; ++c) {
                        const uint8_t* q = raw.data() + p + (i * channels + c) * (bits / 8);
                        if (format == 3 && bits == 32) { float f; std::memcpy(&f, q, 4); sum += f; }
                        else if (bits == 16) { int16_t v = int16_t(q[0] | q[1] << 8); sum += v / 32768.0; }
                        else if (bits == 24) { int32_t v = (q[0] << 8 | q[1] << 16 | q[2] << 24) >> 8; sum += v / 8388608.0; }
                        else if (bits == 32) { int32_t v; std::memcpy(&v, q, 4); sum += v / 2147483648.0; }
                    }
                    out[i] = float(sum / channels);
                }
                break;
            }
            p += len + (len & 1);
        }
        return out;
    }
    rate = givenRate;
    out.resize(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), out.size() * 4);
    return out;
}

static Controls controlsFromJson(const BrainFile& b, const Json& j) {
    Controls c(b);
    if (j.has("lesion")) for (auto& v : j["lesion"].arr) c.lesion[size_t(v.asInt())] = 0.0f;
    if (j.has("ear_mute")) for (auto& v : j["ear_mute"].arr) c.earMute[size_t(v.asInt())] = 0.0f;
    if (j.has("gain_scale")) c.gainScale = j["gain_scale"].asNumber();
    if (j.has("tau_scale")) c.tauScale = j["tau_scale"].asNumber();
    if (j.has("sign_flip")) for (auto& v : j["sign_flip"].arr) c.signFlip[size_t(v.asInt())] = -1.0f;
    if (j.has("unsilence")) for (auto& v : j["unsilence"].arr) c.unsilence[size_t(v.at(0).asInt())] = int8_t(v.at(1).asInt());
    if (j.has("band_offset")) for (size_t k = 0; k < j["band_offset"].size() && k < c.bandOffset.size(); ++k) c.bandOffset[k] = float(j["band_offset"].at(k).asNumber());
    if (j.has("inject")) for (auto& v : j["inject"].arr) {
        Injection inj; inj.neuron = int(v.at(0).asInt()); inj.current = float(v.at(1).asNumber());
        inj.startFrame = v.at(2).asInt(); inj.frames = int(v.at(3).asInt());
        c.addInjection(inj);
    }
    if (j.has("threshold")) c.threshold = j["threshold"].asNumber();
    return c;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: flybrain_render brain.flybrain --audio in.f32 --rate R [options]\n"); return 2; }
    std::string brainPath = argv[1], audioPath, featPath, controlsPath, dumpPath, recordPath;
    int rate = 16000, block = 512, preroll = 0;
    bool dbl = false, ftz = true, feat32 = false;
    double trim = 0.0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--audio") audioPath = next();
        else if (a == "--rate") rate = std::stoi(next());
        else if (a == "--block") block = std::stoi(next());
        else if (a == "--double") dbl = true;
        else if (a == "--no-ftz") ftz = false;
        else if (a == "--preroll") preroll = std::stoi(next());
        else if (a == "--trim") trim = std::stod(next());
        else if (a == "--controls") controlsPath = next();
        else if (a == "--dump") dumpPath = next();
        else if (a == "--record") recordPath = next();
        else if (a == "--features") { featPath = next(); feat32 = false; }
        else if (a == "--features32") { featPath = next(); feat32 = true; }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    try {
        BrainFile brain = BrainFile::load(brainPath);
        Engine engine(brain);
        EngineConfig cfg;
        cfg.hostRate = rate; cfg.maxBlock = block; cfg.doublePrecision = dbl; cfg.flushDenormals = ftz;
        cfg.prerollFrames = preroll; cfg.inputTrimDb = trim;
        FILE* dump = nullptr;
        if (!dumpPath.empty()) {
            dump = std::fopen(dumpPath.c_str(), "wb");
            std::fwrite("FLYDUMP1", 1, 8, dump);
            uint32_t dims[4] = {uint32_t(brain.nBands), uint32_t(brain.n), uint32_t(brain.nMn()), uint32_t(brain.nDrums)};
            std::fwrite(dims, 4, 4, dump);
        }
        std::vector<float> audio;
        std::vector<double> feats;
        size_t nFrames = 0;
        if (!featPath.empty()) {
            std::vector<uint8_t> raw = readFile(featPath);
            size_t per = size_t(brain.nBands);
            if (feat32) { nFrames = raw.size() / (4 * per); feats.resize(nFrames * per); for (size_t i = 0; i < feats.size(); ++i) { float f; std::memcpy(&f, raw.data() + 4 * i, 4); feats[i] = f; } }
            else { nFrames = raw.size() / (8 * per); feats.resize(nFrames * per); std::memcpy(feats.data(), raw.data(), feats.size() * 8); }
            cfg.hostRate = 16000;
        } else {
            audio = loadAudio(audioPath, rate, rate);
            cfg.hostRate = rate;
        }
        engine.prepare(cfg);
        if (!recordPath.empty()) {
            Json info; info.type = Json::Object;
            Json s; s.type = Json::String; s.str = brain.sha256; info.obj["brain_sha256"] = s;
            s.str = brain.name; info.obj["brain_name"] = s;
            Json nnum; nnum.type = Json::Number; nnum.num = cfg.hostRate; info.obj["sample_rate"] = nnum;
            nnum.num = block; info.obj["block_size"] = nnum;
            s.str = dbl ? "double" : "float"; info.obj["precision"] = s;
            s.str = "flybrain_render"; info.obj["host"] = s;
            engine.recorder().start(recordPath, recordPath + ".json", info);
        }
        Json ctlJson;
        if (!controlsPath.empty()) {
            std::vector<uint8_t> t = readFile(controlsPath);
            ctlJson = Json::parse(std::string(t.begin(), t.end()));
            Controls c = controlsFromJson(brain, ctlJson);
            engine.setControls(c, ctlJson.dump());
        }
        Snapshot snap;
        engine.allocateSnapshot(snap);
        std::vector<HitEvent> events(4096);
        std::printf("{\"brain\":\"%s\",\"precision\":\"%s\",\"latency_host_samples\":%d,\"hits\":[", brain.sha256.c_str(), dbl ? "double" : "float", engine.latencyHostSamples());
        bool first = true;
        auto writeDump = [&]() {
            if (!dump) return;
            while (engine.readSnapshot(snap)) {
                std::fwrite(&snap.frame, 8, 1, dump);
                uint8_t flags[4] = {uint8_t(snap.resetFlag), uint8_t(snap.runaway), uint8_t(snap.neutral), uint8_t(snap.muted)};
                std::fwrite(flags, 1, 4, dump);
                std::fwrite(snap.bands.data(), 8, snap.bands.size(), dump); std::fwrite(snap.ear.data(), 8, snap.ear.size(), dump);
                std::fwrite(snap.rates.data(), 8, snap.rates.size(), dump); std::fwrite(snap.mn.data(), 8, snap.mn.size(), dump);
                std::fwrite(snap.prob.data(), 8, snap.prob.size(), dump); std::fwrite(snap.vel.data(), 8, snap.vel.size(), dump);
                std::fwrite(snap.hits.data(), 1, snap.hits.size(), dump);
            }
        };
        auto emit = [&](int n, int64_t blockStart) {
            for (int k = 0; k < n; ++k) {
                std::printf("%s[%lld,%d,%.9g,%lld]", first ? "" : ",", (long long)events[k].frame, events[k].drum, events[k].velocity,
                            events[k].sampleOffset < 0 ? -1LL : (long long)(blockStart + events[k].sampleOffset));
                first = false;
            }
        };
        if (!featPath.empty()) {
            for (size_t f = 0; f < nFrames; ++f) {
                int n = engine.stepFeatures(feats.data() + f * size_t(brain.nBands), events.data(), int(events.size()));
                emit(n, int64_t(f));
                writeDump();   // one frame at a time: the snapshot holds only the newest, so read after each step
            }
        } else {
            std::vector<float> buf(static_cast<size_t>(block), 0.0f);
            const float* chans[1] = {buf.data()};
            for (size_t s = 0; s < audio.size(); s += size_t(block)) {
                int n = int(std::min(size_t(block), audio.size() - s));
                std::memcpy(buf.data(), audio.data() + s, size_t(n) * 4);
                if (dump && block > 80 * rate / 16000) {
                    // the dump needs every frame: feed the block in slices no longer than a frame
                    int slice = std::max(1, 80 * rate / 16000);
                    for (int o = 0; o < n; o += slice) {
                        const float* ch[1] = {buf.data() + o};
                        int m = engine.process(ch, 1, std::min(slice, n - o), events.data(), int(events.size()));
                        emit(m, int64_t(s) + o);
                        writeDump();
                    }
                } else {
                    int m = engine.process(chans, 1, n, events.data(), int(events.size()));
                    emit(m, int64_t(s));
                    writeDump();
                }
            }
        }
        int m = engine.flush(events.data(), int(events.size()));
        emit(m, 0);
        writeDump();
        if (dump) std::fclose(dump);
        if (!recordPath.empty()) engine.recorder().stop();
        std::printf("],\"frames\":%lld,\"runaways\":%d,\"recorder_dropped\":%llu}\n", (long long)engine.framesProcessed(), engine.runawayCount(),
                    (unsigned long long)engine.recorder().framesDropped());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
