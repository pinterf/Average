## Average ##

A simple plugin that calculates weighted average of multiple frames.

### Usage
```
Average(clip1, weight1, clip2, weight2, clip3, weight3, ...)
```
The usage is identical to the old Average plugin or RedAverage.
Output pixel value is calculated as 
```
out[x] = clip1[x] * weight1 + clip2[x] * weight2 + clip3[x] * weight3...
```
The filter performs faster when all weight in the call are less or equal to one. This filter should be faster than the old Average and more stable (alas slower) than RedAverage.

### History
```
v0.94 (20170127)
Fix: fix the fix: rounding of intermediate results was ok for two clips
New: AVX for 10-16bit (+20-30%) and float (+50-60%) compared to v0.93
     AVX for 8 bit non-integer path (+20% gain), e.g. when one of the weights is over 1.0
     Note 1: AVX needs 32 byte frame alignment (Avisynth+ default)
     Note 2: AVX CPU flag is reported by recent Avisynth+ version
     Note 3: AVX is reported only on approriate OS (from Windows 7 SP1 on)

v0.93 (20170126 - pinterf)
Fix: rounding of intermediate results in fast integer average of 8 bit clips
Mod: faster results for two or three clips
New: Support for Avisynth+ color spaces: 10-16 bit and float YUV(A)/Planar RGB(A), RGB48 and RGB64
     10+ bits are calculated in float precision internally.
New: auto register as NICE_FILTER for Avisynth+
New: add version resource
Info: built with VS2015 Update 3, may require Visual Studio 2015 Redistributable update 3

v0.92 (20131227 - tp7) 
This release fixes a very important memory leak which made the plugin unusable for somewhat complex scripts.

v0.91 (20131224 - tp7)
Double performance when absolute values of all weights are smaller or equal to one.

v0.90 (20131221 - tp7)
Initial release.
```

