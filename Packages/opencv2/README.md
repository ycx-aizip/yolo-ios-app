# OpenCV Package for AizipFishCount

This package contains OpenCV frameworks and the Objective-C++ bridge required by the AizipFishCount backend.

## Contents

```
OpenCV/
├── ios/                  # OpenCV framework for iOS devices
├── ios-simulator/        # OpenCV framework for iOS simulator
├── OpenCVBridge.h        # Objective-C++ bridge header
├── OpenCVBridge.mm       # Objective-C++ bridge implementation
└── README.md            # This file
```

## Usage Instructions

### For Your Xcode Project (Frontend)

The AizipFishCount backend uses OpenCV through the OpenCVBridge. To integrate:

1. **Add OpenCVBridge files to your Xcode project:**
   - Drag `OpenCVBridge.h` and `OpenCVBridge.mm` into your Xcode project
   - Make sure "Copy items if needed" is **unchecked** (reference the files)
   - Ensure they are added to your app target

2. **Import in your Bridging Header:**
   ```objc
   #import "OpenCVBridge.h"
   ```

3. **Configure Framework Search Paths:**
   - Add framework search path to OpenCV frameworks:
     - Simulator: `$(SRCROOT)/../Packages/OpenCV/ios-simulator`
     - Device: `$(SRCROOT)/../Packages/OpenCV/ios`

4. **Link against OpenCV:**
   - The build system will automatically link `opencv2.framework`

## Why These Files Are Here

- **OpenCVBridge.h/.mm**: Provides Objective-C++ bridge between Swift and OpenCV C++ API
- The AizipFishCount Swift package accesses these dynamically via Objective-C runtime
- Users must compile these files as part of their app target (Swift Package Manager doesn't support .mm files)

## For Partners Using xcframework

When distributing AizipFishCount.xcframework to partners:

1. Include this entire `Packages/OpenCV/` directory
2. Provide setup instructions (as above)
3. Partners must add `OpenCVBridge.h/.mm` to their Xcode project
4. Partners must configure framework search paths

This approach keeps OpenCV integration flexible and allows partners to use their own OpenCV version if needed.
