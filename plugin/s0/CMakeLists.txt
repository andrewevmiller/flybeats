# S0: two trivial VST3 plugins whose only job is to answer the five FL Studio observations
# (plugin/s0/OBSERVATIONS.md). Nothing here is the flybeats engine.
cmake_minimum_required(VERSION 3.22)
project(FlyS0 VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# macOS: native arm64 so FL Studio does not bridge the plugin (plan, decision D3); universal on request.
option(FLYS0_UNIVERSAL "macOS: build a universal (arm64 + x86_64) binary" OFF)
if(APPLE)
    if(FLYS0_UNIVERSAL)
        set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "" FORCE)
    else()
        set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "" FORCE)
    endif()
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "" FORCE)
endif()

# JUCE 8.0.8 (the VST3 SDK ships inside it). A checkout already at plugin/s0/_deps/JUCE is used as is,
# otherwise CMake fetches it from GitHub.
include(FetchContent)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/_deps/JUCE/CMakeLists.txt" AND NOT DEFINED FETCHCONTENT_SOURCE_DIR_JUCE)
    set(FETCHCONTENT_SOURCE_DIR_JUCE "${CMAKE_CURRENT_SOURCE_DIR}/_deps/JUCE" CACHE PATH "" FORCE)
endif()
FetchContent_Declare(JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.8
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(JUCE)

# Common settings for both targets. No signing here: the owner signs and notarises (plan, section 3).
function(flys0_common target)
    target_compile_definitions(${target} PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0   # no VST2 identity, we never shipped one
        JUCE_DISPLAY_SPLASH_SCREEN=0)  # allowed under the JUCE 8 licence the owner picks (plan, D2); set to 1 if on Starter terms that require it
    target_link_libraries(${target} PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)
endfunction()

# --- Fly S0 Effect: mixer-insert effect with main in/out, a sidechain input, MIDI out and a reported latency
juce_add_plugin(FlyS0Effect
    COMPANY_NAME "flybeats"
    BUNDLE_ID "com.flybeats.flys0effect"
    PLUGIN_MANUFACTURER_CODE Flyb
    PLUGIN_CODE Fs0e
    PRODUCT_NAME "Fly S0 Effect"
    FORMATS VST3
    VST3_CATEGORIES Fx Analyzer
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT TRUE            # observation (3): does FL pass MIDI out from an effect?
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE)
target_sources(FlyS0Effect PRIVATE Source/EffectProcessor.cpp)
target_include_directories(FlyS0Effect PRIVATE Source)
flys0_common(FlyS0Effect)

# --- Fly S0 Instrument: a generator with audio out, MIDI out and an optional audio input bus
juce_add_plugin(FlyS0Instrument
    COMPANY_NAME "flybeats"
    BUNDLE_ID "com.flybeats.flys0instrument"
    PLUGIN_MANUFACTURER_CODE Flyb
    PLUGIN_CODE Fs0i
    PRODUCT_NAME "Fly S0 Instrument"
    FORMATS VST3
    VST3_CATEGORIES Instrument Synth
    IS_SYNTH TRUE                      # observation (2): a generator channel; does it get mixer audio?
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT TRUE             # observation (3): MIDI out from an instrument
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD FALSE)
target_sources(FlyS0Instrument PRIVATE Source/InstrumentProcessor.cpp)
target_include_directories(FlyS0Instrument PRIVATE Source)
flys0_common(FlyS0Instrument)
