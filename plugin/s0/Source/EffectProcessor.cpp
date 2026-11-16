// Fly S0 Effect: a VST3 *effect* (mixer-insert) whose only purpose is to answer, in FL Studio:
//   (1) does an effect on a mixer insert hear other tracks (routed into the insert, and through the sidechain input)?
//   (3) does FL route MIDI out from an effect?
//   (4) does FL's delay compensation follow the latency the plugin reports?
// It meters its main and sidechain inputs, clicks (audio) and sends a MIDI note on crude onsets of the chosen input,
// and passes the main input through delayed by the latency it reports.
#include "Common.h"

namespace flys0
{

class EffectProcessor : public juce::AudioProcessor
{
public:
    EffectProcessor()
        // Bus layout: a stereo main input, a stereo main output and a second stereo input. JUCE's VST3 wrapper
        // declares any input bus after the first as an auxiliary (sidechain) bus; FL Studio's wrapper then lets the
        // user assign mixer tracks to it in the Processing tab. This is the layout the main plugin will use
        // (plan, section 3), so S0 tests exactly it.
        : AudioProcessor (BusesProperties()
                              .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                              .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                              .withInput ("Sidechain", juce::AudioChannelSet::stereo(), true)),
          params (*this, nullptr, "params", makeLayout())
    {
        listen    = params.getRawParameterValue ("listen");
        threshold = params.getRawParameterValue ("threshold");
        latency   = params.getRawParameterValue ("latency");
    }

    static juce::AudioProcessorValueTreeState::ParameterLayout makeLayout()
    {
        using namespace juce;
        AudioProcessorValueTreeState::ParameterLayout l;
        l.add (std::make_unique<AudioParameterChoice> (ParameterID { "listen", 1 }, "Listen",
                                                       StringArray { "Main input", "Sidechain input" }, 0));
        l.add (std::make_unique<AudioParameterFloat> (ParameterID { "threshold", 1 }, "Onset threshold (dB)",
                                                      NormalisableRange<float> (-60.0f, 0.0f, 0.1f), -30.0f));
        // Reported latency in ms. Observation (4): with compensation on, the delayed pass-through (and the clicks) stay
        // on the grid whatever this says; with it off, they move late by exactly this amount.
        l.add (std::make_unique<AudioParameterChoice> (ParameterID { "latency", 1 }, "Reported latency",
                                                       StringArray { "0 ms", "50 ms", "100 ms", "200 ms" }, 0));
        return l;
    }

    // Accept stereo or mono on each bus; the sidechain may be disabled (hosts without one still load the plugin).
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override
    {
        auto ok = [] (const juce::AudioChannelSet& s) { return s == juce::AudioChannelSet::stereo() || s == juce::AudioChannelSet::mono(); };
        if (! ok (layouts.getMainInputChannelSet()) || ! ok (layouts.getMainOutputChannelSet())) return false;
        const auto sc = layouts.getChannelSet (true, 1);
        return sc.isDisabled() || ok (sc);
    }

    void prepareToPlay (double sampleRate, int samplesPerBlock) override
    {
        sr = sampleRate;
        click.prepare (sampleRate);
        rms.prepare (sampleRate, 0.020);              // the 20 ms onset window
        refractorySamples = (int) (0.100 * sampleRate);
        maxDelay = (int) std::ceil (0.200 * sampleRate) + samplesPerBlock + 1;
        delayLine.setSize (2, maxDelay);
        delayLine.clear();
        writePos = 0;
        applyLatency();
    }
    void releaseResources() override {}

    int latencySamplesFor (int choice) const { return (int) std::round (latencyMs[choice] * 0.001 * sr); }

    // The reported latency: setLatencySamples tells the host (JUCE's VST3 wrapper asks the host to re-read it).
    void applyLatency()
    {
        const int choice = (int) latency->load();
        currentDelay = latencySamplesFor (choice);
        if (getLatencySamples() != currentDelay)
            setLatencySamples (currentDelay);
    }

    using juce::AudioProcessor::processBlock;   // the double-precision overload stays JUCE's default
    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) override
    {
        juce::ScopedNoDenormals noDenormals;
        applyLatency();
        const int n = buffer.getNumSamples();
        auto mainIn = getBusBuffer (buffer, true, 0);
        const bool haveSide = getBus (true, 1) != nullptr && getBus (true, 1)->isEnabled();
        auto sideIn = haveSide ? getBusBuffer (buffer, true, 1) : juce::AudioBuffer<float>();
        mainMeter.update (mainIn);
        if (haveSide) sideMeter.update (sideIn); else sideMeter.peak.store (0.0f);
        sidechainEnabled.store (haveSide);

        // Copy the main input into the delay line before anything writes to the buffer (in-place processing).
        for (int i = 0; i < n; ++i)
        {
            for (int ch = 0; ch < 2; ++ch)
            {
                const int src = juce::jmin (ch, mainIn.getNumChannels() - 1);
                delayLine.setSample (ch, writePos, src >= 0 ? mainIn.getSample (src, i) : 0.0f);
            }
            if (++writePos >= maxDelay) writePos = 0;
        }

        // Onset detection on the chosen input's mono mean (mean, not sum: the training mix averages channels).
        const bool useSide = (int) listen->load() == 1;
        const auto& src = useSide ? sideIn : mainIn;
        const int srcCh = src.getNumChannels();
        const float thr = juce::Decibels::decibelsToGain (threshold->load());
        midi.clear();                                    // the effect takes no MIDI in; everything below is output
        for (int i = 0; i < n; ++i)
        {
            float x = 0.0f;
            for (int ch = 0; ch < srcCh; ++ch) x += src.getSample (ch, i);
            if (srcCh > 0) x /= (float) srcCh;
            const float level = rms.push (x);
            const bool above = level >= thr;
            if (above && ! wasAbove && sinceOnset >= refractorySamples)
            {
                // Observation (3): a note on the plugin's MIDI output at the onset's sample position.
                const auto vel = (juce::uint8) juce::jlimit (1, 127, (int) (127.0f * juce::jmin (1.0f, level * 4.0f)));
                midi.addEvent (juce::MidiMessage::noteOn (1, 36, vel), i);
                pendingNoteOff = (int) (0.010 * sr);     // note off 10 ms later
                click.trigger (0.5f);
                sinceOnset = 0;
                onsets.fetch_add (1);
            }
            wasAbove = above;
            ++sinceOnset;
            if (pendingNoteOff > 0 && --pendingNoteOff == 0)
                midi.addEvent (juce::MidiMessage::noteOff (1, 36), i);
        }

        // Output: main input delayed by the reported latency, plus the clicks (undelayed, at the moment of decision).
        auto out = getBusBuffer (buffer, false, 0);
        int readPos = writePos - n - currentDelay;
        while (readPos < 0) readPos += maxDelay;
        for (int i = 0; i < n; ++i)
        {
            const float c = click.next();
            for (int ch = 0; ch < out.getNumChannels(); ++ch)
                out.setSample (ch, i, delayLine.getSample (juce::jmin (ch, 1), readPos) + c);
            if (++readPos >= maxDelay) readPos = 0;
        }
    }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Fly S0 Effect"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return true; }     // observation (3)
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock& dest) override
    {
        if (auto xml = params.copyState().createXml()) copyXmlToBinary (*xml, dest);
    }
    void setStateInformation (const void* data, int size) override
    {
        if (auto xml = getXmlFromBinary (data, size)) params.replaceState (juce::ValueTree::fromXml (*xml));
    }

    juce::AudioProcessorValueTreeState params;
    Meter mainMeter, sideMeter;
    std::atomic<bool> sidechainEnabled { false };
    std::atomic<int> onsets { 0 };

private:
    std::atomic<float>* listen = nullptr;
    std::atomic<float>* threshold = nullptr;
    std::atomic<float>* latency = nullptr;
    static constexpr double latencyMs[4] = { 0.0, 50.0, 100.0, 200.0 };
    double sr = 48000.0;
    Click click;
    RmsWindow rms;
    bool wasAbove = false;
    int sinceOnset = 1 << 30, refractorySamples = 4800, pendingNoteOff = 0;
    juce::AudioBuffer<float> delayLine;
    int maxDelay = 1, writePos = 0, currentDelay = 0;
};

// The editor: JUCE's generic parameter panel plus the two input meters, refreshed 20 times a second.
class EffectEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit EffectEditor (EffectProcessor& p) : AudioProcessorEditor (p), proc (p), generic (p)
    {
        addAndMakeVisible (generic);
        addAndMakeVisible (mainBar);
        addAndMakeVisible (sideBar);
        addAndMakeVisible (info);
        info.setColour (juce::Label::textColourId, juce::Colours::white);
        setSize (520, 260);
        startTimerHz (20);
    }
    void paint (juce::Graphics& g) override { g.fillAll (juce::Colour (0xff202020)); }
    void resized() override
    {
        auto r = getLocalBounds().reduced (8);
        mainBar.setBounds (r.removeFromTop (24));
        sideBar.setBounds (r.removeFromTop (24));
        info.setBounds (r.removeFromTop (24));
        generic.setBounds (r);
    }
private:
    void timerCallback() override
    {
        mainBar.set (proc.mainMeter.peak.load());
        sideBar.set (proc.sideMeter.peak.load());
        info.setText (juce::String ("sidechain bus: ") + (proc.sidechainEnabled.load() ? "enabled" : "disabled")
                      + "   onsets sent as MIDI note 36: " + juce::String (proc.onsets.load())
                      + "   reported latency: " + juce::String (proc.getLatencySamples()) + " samples",
                      juce::dontSendNotification);
    }
    EffectProcessor& proc;
    juce::GenericAudioProcessorEditor generic;
    MeterBar mainBar { "main input" }, sideBar { "sidechain input" };
    juce::Label info;
};

juce::AudioProcessorEditor* EffectProcessor::createEditor() { return new EffectEditor (*this); }

} // namespace flys0

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new flys0::EffectProcessor(); }
