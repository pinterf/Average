#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#endif

#include <algorithm>
#include <avisynth.h>
#include <avs/alignment.h>
#include <immintrin.h>
#include <stdint.h>
#include <vector>

template<typename pixel_t, int bits_per_pixel>
void weighted_average_avx(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height);

void weighted_average_f_avx(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height);

