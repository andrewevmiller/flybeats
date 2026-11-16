// A tiny assert framework for the engine tests.
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static int g_failures = 0, g_checks = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)
#define CHECK_MSG(cond, ...) do { ++g_checks; if (!(cond)) { ++g_failures; std::printf("FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
#define REQUIRE(cond) do { ++g_checks; if (!(cond)) { ++g_failures; std::printf("FATAL %s:%d: %s\n", __FILE__, __LINE__, #cond); return finish(); } } while (0)
static int finish() {
    std::printf("%s: %d checks, %d failures\n", g_failures ? "FAILED" : "passed", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
static bool throwsWith(const std::function<void()>& f, const std::string& needle) {
    try { f(); } catch (const std::exception& e) { return std::string(e.what()).find(needle) != std::string::npos; }
    return false;
}
