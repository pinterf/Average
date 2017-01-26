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
v0.93 (20170126 - pinterf)
Fix: rounding of intermediate results in fast integer average of 8 bit clips
Mod: faster results for two or three clips
New: Support for Avisynth+ color spaces: 10-16 bit and float YUV(A)/Planar RGB(A), RGB48 and RGB64
     10+ bits are calculated in float precision internally.
New: auto register as NICE_FILTER for Avisynth+
New: add version resource
Info: built with VS2015 Update 3, may require Visual Studio 2015 Redistributable update 3

v0.92 (20141227 - tp7) 
This release fixes a very important memory leak which made the plugin unusable for somewhat complex scripts.

v0.91 (20141224 - tp7)
Double performance when absolute values of all weights are smaller or equal to one.

v0.90 (20141221 - tp7)
Initial release.
```

