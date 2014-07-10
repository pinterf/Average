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
