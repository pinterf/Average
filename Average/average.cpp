#include "average_avx.h"

template<int minimum, int maximum>
static AVS_FORCEINLINE int static_clip(float val) {
    if (val > maximum) {
        return maximum;
    }
    if (val < minimum) {
        return minimum;
    }
    return (int)val;
}

template<typename pixel_t, int bits_per_pixel>
static inline void weighted_average_c(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
  // width is rowsize
  const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : ((1 << bits_per_pixel) - 1);

  width /= sizeof(pixel_t);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float acc = 0;
      for (int i = 0; i < frames_count; ++i) {
        acc += reinterpret_cast<const pixel_t *>(src_pointers[i])[x] * weights[i];
      }
      if (sizeof(pixel_t) == 4)
        reinterpret_cast<float *>(dstp)[x] = acc;
      else
        reinterpret_cast<pixel_t *>(dstp)[x] = (pixel_t)(static_clip<0, max_pixel_value>(acc));
    }

    for (int i = 0; i < frames_count; ++i) {
      src_pointers[i] += src_pitches[i];
    }
    dstp += dst_pitch;
  }
}

// fake _mm_packus_epi32 (orig is SSE4.1 only)
AVS_FORCEINLINE __m128i _MM_PACKUS_EPI32(__m128i a, __m128i b)
{
  a = _mm_slli_epi32(a, 16);
  a = _mm_srai_epi32(a, 16);
  b = _mm_slli_epi32(b, 16);
  b = _mm_srai_epi32(b, 16);
  a = _mm_packs_epi32(a, b);
  return a;
}

__m128i compile_safe_mm_packus_epi32 (__m128i a, __m128i b) {
    #ifdef __SSE4_1__
    return _mm_packus_epi32(a, b);
    #elif __SSE4_2__
    return _mm_packus_epi32(a, b);
    #else
    return _MM_PACKUS_EPI32(a, b); // SSE2 friendly but slower
    #endif
}

template<typename pixel_t, int bits_per_pixel>
static inline void weighted_average_sse2(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
    // width is row_size
    int mod_width;
    if(sizeof(pixel_t) == 1)
      mod_width = width / 8 * 8;
    else
      mod_width = width / 16 * 16;

    const int sse_size = (sizeof(pixel_t) == 1) ? 8 : 16;

    const int max_pixel_value = (sizeof(pixel_t) == 1) ? 255 : ((1 << bits_per_pixel) - 1);
    __m128i pixel_limit;
    if (sizeof(pixel_t) == 2 && bits_per_pixel < 16)
      pixel_limit = _mm_set1_epi16((int16_t)max_pixel_value);

    __m128i zero = _mm_setzero_si128();

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < mod_width; x += sse_size) {
            __m128 acc_lo = _mm_setzero_ps();
            __m128 acc_hi = _mm_setzero_ps();
            
            for (int i = 0; i < frames_count; ++i) {
                __m128i src;
                if (sizeof(pixel_t) == 1)
                  src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[i] + x));
                else
                  src = _mm_load_si128(reinterpret_cast<const __m128i*>(src_pointers[i] + x));
                auto weight = _mm_set1_ps(weights[i]);

                if(sizeof(pixel_t) == 1)
                  src = _mm_unpacklo_epi8(src, zero);
                auto src_lo_ps = _mm_cvtepi32_ps(_mm_unpacklo_epi16(src, zero));
                auto src_hi_ps = _mm_cvtepi32_ps(_mm_unpackhi_epi16(src, zero));

                auto weighted_lo = _mm_mul_ps(src_lo_ps, weight);
                auto weighted_hi = _mm_mul_ps(src_hi_ps, weight);
                
                acc_lo = _mm_add_ps(acc_lo, weighted_lo);
                acc_hi = _mm_add_ps(acc_hi, weighted_hi);
            }
            auto dst_lo = _mm_cvtps_epi32(acc_lo);
            auto dst_hi = _mm_cvtps_epi32(acc_hi);

            __m128i dst;

            if (sizeof(pixel_t) == 1) {
              dst = _mm_packs_epi32(dst_lo, dst_hi);
              dst = _mm_packus_epi16(dst, zero);
            }
            else if (sizeof(pixel_t) == 2) {
              if (bits_per_pixel < 16) {
                dst = _mm_packs_epi32(dst_lo, dst_hi); // no need for packus
              } else {
                dst = compile_safe_mm_packus_epi32(dst_lo, dst_hi);
              }
            }
            
            if (sizeof(pixel_t) == 2 && bits_per_pixel < 16)
              dst = _mm_min_epi16(dst, pixel_limit); // no need for SSE4 epu16 

            if(sizeof(pixel_t) == 1)
              _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp+x), dst);
            else
              _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x), dst);
        }

        int start = mod_width / sizeof(pixel_t);
        int end = width / sizeof(pixel_t);
        for (int x = start; x < end; ++x) {
            float acc = 0;
            for (int i = 0; i < frames_count; ++i) {
                acc += reinterpret_cast<const pixel_t *>(src_pointers[i])[x] * weights[i];
            }
            reinterpret_cast<pixel_t *>(dstp)[x] = static_clip<0, max_pixel_value>(acc);
        }

        for (int i = 0; i < frames_count; ++i) {
            src_pointers[i] += src_pitches[i];
        }
        dstp += dst_pitch;
    }
}

static inline void weighted_average_f_sse2(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
  // width is row_size
  int mod_width = width / 16 * 16;

  const int sse_size = 16;

  __m128i zero = _mm_setzero_si128();

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < mod_width; x += sse_size) {
      __m128 acc = _mm_setzero_ps();

      for (int i = 0; i < frames_count; ++i) {
        __m128 src;
        src = _mm_load_ps(reinterpret_cast<const float*>(src_pointers[i] + x));
        auto weight = _mm_set1_ps(weights[i]);

        auto weighted = _mm_mul_ps(src, weight);

        acc = _mm_add_ps(acc, weighted);
      }

      _mm_store_ps(reinterpret_cast<float*>(dstp + x), acc);
    }

    for (int x = mod_width / 4; x < width / 4; ++x) {
      float acc = 0;
      for (int i = 0; i < frames_count; ++i) {
        acc += reinterpret_cast<const float *>(src_pointers[i])[x] * weights[i];
      }
      reinterpret_cast<float *>(dstp)[x] = acc; // float: no clamping
    }

    for (int i = 0; i < frames_count; ++i) {
      src_pointers[i] += src_pitches[i];
    }
    dstp += dst_pitch;
  }
}


template<int frames_count_2_3_more>
static inline void weighted_average_int_sse2(uint8_t *dstp, int dst_pitch, const uint8_t **src_pointers, int *src_pitches, float *weights, int frames_count, int width, int height) {
    int16_t *int_weights = reinterpret_cast<int16_t*>(alloca(frames_count*sizeof(int16_t)));
    for (int i = 0; i < frames_count; ++i) {
        int_weights[i] = static_cast<int16_t>((1 << 14) * weights[i]);
    }
    int mod8_width = width / 8 * 8;
    __m128i zero = _mm_setzero_si128();

    __m128i round_mask = _mm_set1_epi32(0x2000);

    bool even_frames = (frames_count % 2 != 0);

    if (frames_count_2_3_more == 2 || frames_count_2_3_more == 3) {
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < mod8_width; x += 8) {
          __m128i acc_lo = _mm_setzero_si128();
          __m128i acc_hi = _mm_setzero_si128();

          __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[0] + x));
          __m128i src2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[1] + x));
          __m128i weight = _mm_set1_epi32(*reinterpret_cast<int*>(int_weights));

          src = _mm_unpacklo_epi8(src, zero);
          src2 = _mm_unpacklo_epi8(src2, zero);
          __m128i src_lo = _mm_unpacklo_epi16(src, src2);
          __m128i src_hi = _mm_unpackhi_epi16(src, src2);

          __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
          __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

          acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
          acc_hi = _mm_add_epi32(acc_hi, weighted_hi);

          if (frames_count_2_3_more == 3) {
            __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[2] + x));
            __m128i weight = _mm_set1_epi32(int_weights[2]);

            src = _mm_unpacklo_epi8(src, zero);
            __m128i src_lo = _mm_unpacklo_epi16(src, zero);
            __m128i src_hi = _mm_unpackhi_epi16(src, zero);

            __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
            __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

            acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
            acc_hi = _mm_add_epi32(acc_hi, weighted_hi);
          }

          acc_lo = _mm_add_epi32(acc_lo, round_mask);
          acc_hi = _mm_add_epi32(acc_hi, round_mask);

          __m128i dst_lo = _mm_srai_epi32(acc_lo, 14);
          __m128i dst_hi = _mm_srai_epi32(acc_hi, 14);

          __m128i dst = _mm_packs_epi32(dst_lo, dst_hi);
          dst = _mm_packus_epi16(dst, zero);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), dst);
        }

        for (int x = mod8_width; x < width; ++x) {
          float acc = 0;
          acc += src_pointers[0][x] * weights[0];
          acc += src_pointers[1][x] * weights[1];
          if (frames_count_2_3_more == 3)
            acc += src_pointers[2][x] * weights[2];
          dstp[x] = static_clip<0, 255>(acc);
        }
       
        src_pointers[0] += src_pitches[0];
        src_pointers[1] += src_pitches[1];
        if (frames_count_2_3_more == 3)
          src_pointers[2] += src_pitches[2];
        dstp += dst_pitch;
      }
    } else {
      // generic path
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < mod8_width; x += 8) {
          __m128i acc_lo = _mm_setzero_si128();
          __m128i acc_hi = _mm_setzero_si128();

          for (int i = 0; i < frames_count - 1; i += 2) {
            __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[i] + x));
            __m128i src2 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[i + 1] + x));
            __m128i weight = _mm_set1_epi32(*reinterpret_cast<int*>(int_weights + i));

            src = _mm_unpacklo_epi8(src, zero);
            src2 = _mm_unpacklo_epi8(src2, zero);
            __m128i src_lo = _mm_unpacklo_epi16(src, src2);
            __m128i src_hi = _mm_unpackhi_epi16(src, src2);

            __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
            __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

            acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
            acc_hi = _mm_add_epi32(acc_hi, weighted_hi);
          }

          if (even_frames) {
            __m128i src = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_pointers[frames_count - 1] + x));
            __m128i weight = _mm_set1_epi32(int_weights[frames_count - 1]);

            src = _mm_unpacklo_epi8(src, zero);
            __m128i src_lo = _mm_unpacklo_epi16(src, zero);
            __m128i src_hi = _mm_unpackhi_epi16(src, zero);

            __m128i weighted_lo = _mm_madd_epi16(src_lo, weight);
            __m128i weighted_hi = _mm_madd_epi16(src_hi, weight);

            acc_lo = _mm_add_epi32(acc_lo, weighted_lo);
            acc_hi = _mm_add_epi32(acc_hi, weighted_hi);
          }

          acc_lo = _mm_add_epi32(acc_lo, round_mask);
          acc_hi = _mm_add_epi32(acc_hi, round_mask);

          __m128i dst_lo = _mm_srai_epi32(acc_lo, 14);
          __m128i dst_hi = _mm_srai_epi32(acc_hi, 14);

          __m128i dst = _mm_packs_epi32(dst_lo, dst_hi);
          dst = _mm_packus_epi16(dst, zero);
          _mm_storel_epi64(reinterpret_cast<__m128i*>(dstp + x), dst);
        }

        for (int x = mod8_width; x < width; ++x) {
          float acc = 0;
          for (int i = 0; i < frames_count; ++i) {
            acc += src_pointers[i][x] * weights[i];
          }
          dstp[x] = static_clip<0, 255>(acc);
        }

        for (int i = 0; i < frames_count; ++i) {
          src_pointers[i] += src_pitches[i];
        }
        dstp += dst_pitch;
      }
    }
}

struct WeightedClip {
    PClip clip;
    float weight;

    WeightedClip(PClip _clip, float _weight) : clip(_clip), weight(_weight) {}
};


class Average : public GenericVideoFilter {
public:
  Average(std::vector<WeightedClip> clips, IScriptEnvironment* env)
    : GenericVideoFilter(clips[0].clip), clips_(clips) {

    int frames_count = (int)clips_.size();

    int pixelsize = vi.ComponentSize();
    int bits_per_pixel = vi.BitsPerComponent();

    bool avx = !!(env->GetCPUFlags() & CPUF_AVX);
    // we don't know the alignment here. avisynth+: 32 bytes, classic: 16
    // decide later (processor_, processor_32aligned)

    if (env->GetCPUFlags() & CPUF_SSE2) {
      bool use_weighted_average_f = false;
      if (pixelsize == 1) {
        if (frames_count == 2)
          processor_ = &weighted_average_int_sse2<2>; 
        else if (frames_count == 3)
          processor_ = &weighted_average_int_sse2<3>;
        else
          processor_ = &weighted_average_int_sse2<0>;
        processor_32aligned_ = processor_;
        for (const auto& clip : clips) {
          if (std::abs(clip.weight) > 1) {
            use_weighted_average_f = true;
            break;
          }
        }
        if (clips.size() > 255) {
          // too many clips, may overflow
          use_weighted_average_f = true;
        }
      }
      else {
        // uint16 and float: float mode internally
        use_weighted_average_f = true;
      }

      if (use_weighted_average_f) {
        switch(bits_per_pixel) {
        case 8: 
          processor_ = &weighted_average_sse2<uint8_t, 8>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint8_t, 8> : processor_;
          break;
        case 10: 
          processor_ = &weighted_average_sse2<uint16_t, 10>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 10> : processor_;
          break;
        case 12:
          processor_ = &weighted_average_sse2<uint16_t, 12>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 12> : processor_;
          break;
        case 14:
          processor_ = &weighted_average_sse2<uint16_t, 14>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 14> : processor_;
          break;
        case 16:
          processor_ = &weighted_average_sse2<uint16_t, 16>;
          processor_32aligned_ = avx ? &weighted_average_avx<uint16_t, 16> : processor_;
          break;
        case 32:
          processor_ = &weighted_average_f_sse2;
          processor_32aligned_ = avx ? &weighted_average_f_avx : processor_;
          break;
        }
      }
    }
    else {
      switch (bits_per_pixel) {
      case 8:
        processor_ = &weighted_average_c<uint8_t, 8>;
        break;
      case 10:
        processor_ = &weighted_average_c<uint16_t, 10>;
        break;
      case 12:
        processor_ = &weighted_average_c<uint16_t, 12>;
        break;
      case 14:
        processor_ = &weighted_average_c<uint16_t, 14>;
        break;
      case 16:
        processor_ = &weighted_average_c<uint16_t, 16>;
        break;
      case 32:
        processor_ = &weighted_average_c<float, 1>; // bits_per_pixel n/a
        break;
      }
      processor_32aligned_ = processor_;
    }
  }

  PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment *env);

  int __stdcall SetCacheHints(int cachehints, int frame_range) override {
    return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
  }

private:
  std::vector<WeightedClip> clips_;
  decltype(&weighted_average_c<uint8_t,8>) processor_;
  decltype(&weighted_average_c<uint8_t, 8>) processor_32aligned_;
};


PVideoFrame Average::GetFrame(int n, IScriptEnvironment *env) {
    int frames_count = (int)clips_.size();
    PVideoFrame* src_frames = reinterpret_cast<PVideoFrame*>(alloca(frames_count * sizeof(PVideoFrame)));
    const uint8_t **src_ptrs = reinterpret_cast<const uint8_t **>(alloca(sizeof(uint8_t*)* frames_count));
    int *src_pitches = reinterpret_cast<int*>(alloca(sizeof(int)* frames_count));
    float *weights = reinterpret_cast<float*>(alloca(sizeof(float)* frames_count));
    if (src_pitches == nullptr || src_frames == nullptr || src_ptrs == nullptr || weights == nullptr) {
        env->ThrowError("Average: Couldn't allocate memory on stack. This is a bug, please report");
    }
    memset(src_frames, 0, frames_count * sizeof(PVideoFrame));

    for (int i = 0; i < frames_count; ++i) {
        src_frames[i] = clips_[i].clip->GetFrame(n, env);
        weights[i] = clips_[i].weight;
    }

    PVideoFrame dst = env->NewVideoFrame(vi);

    int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    int *planes = (vi.IsPlanarRGB() || vi.IsPlanarRGBA()) ? planes_r : planes_y;
    
    bool hasAlpha = vi.IsPlanarRGBA() || vi.IsYUVA();

    for (int pid = 0; pid < (vi.IsY() ? 1 : (hasAlpha ? 4 : 3)); pid++) {
        int plane = planes[pid];
        int width = dst->GetRowSize(plane);
        int height = dst->GetHeight(plane);
        auto dstp = dst->GetWritePtr(plane);
        int dst_pitch = dst->GetPitch(plane);

        bool allSrc32aligned = true;
        for (int i = 0; i < frames_count; ++i) {
            src_ptrs[i] = src_frames[i]->GetReadPtr(plane);
            src_pitches[i] = src_frames[i]->GetPitch(plane);
            if (!IsPtrAligned(src_ptrs[i], 32))
              allSrc32aligned = false;
            if(src_pitches[i] & 0x1F)
              allSrc32aligned = false;
        }

        if(IsPtrAligned(dstp,32) && (dst_pitch & 0x1F)==0 && allSrc32aligned) {
          processor_32aligned_(dstp, dst_pitch, src_ptrs, src_pitches, weights, frames_count, width, height);
        } else {
          processor_(dstp, dst_pitch, src_ptrs, src_pitches, weights, frames_count, width, height);
        }
    }

    for (int i = 0; i < frames_count; ++i) {
        src_frames[i].~PVideoFrame();
    }

    return dst;
}


AVSValue __cdecl create_average(AVSValue args, void* user_data, IScriptEnvironment* env) {
    int arguments_count = args[0].ArraySize();
    if (arguments_count % 2 != 0) {
        env->ThrowError("Average requires an even number of arguments.");
    }
    if (arguments_count == 0) {
        env->ThrowError("Average: At least one clip has to be supplied.");
    }
    std::vector<WeightedClip> clips;
    auto first_clip = args[0][0].AsClip();
    auto first_vi = first_clip->GetVideoInfo();
    clips.emplace_back(first_clip, static_cast<float>(args[0][1].AsFloat()));

    for (int i = 2; i < arguments_count; i += 2) {
        auto clip = args[0][i].AsClip();
        float weight = static_cast<float>(args[0][i+1].AsFloat());
        if (std::abs(weight) < 0.00001f) {
            continue;
        }
        auto vi = clip->GetVideoInfo();
        if (!vi.IsSameColorspace(first_vi)) {
            env->ThrowError("Average: all clips must have the same colorspace.");
        }
        if (vi.width != first_vi.width || vi.height != first_vi.height) {
            env->ThrowError("Average: all clips must have identical width and height.");
        }
        if (vi.num_frames < first_vi.num_frames) {
            env->ThrowError("Average: all clips must be have same or greater number of frames as the first one.");
        }

        clips.emplace_back(clip, weight);
    }

    return new Average(clips, env);
}

const AVS_Linkage *AVS_linkage = nullptr;

extern "C" __declspec(dllexport) const char* __stdcall AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors) {
    AVS_linkage = vectors;
    env->AddFunction("average", ".*", create_average, 0);
    return "Mind your sugar level";
}
