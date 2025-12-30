# iOS Deployment Rules

## Structure

```
yolo-ios-app/                    # Development repo
├── Sources/AizipFishCount/      # Backend (xcframework sources)
│   ├── Predictors/              # TrackingDetector
│   ├── Tracking/                # ByteTracker, STrack, KalmanFilter
│   ├── FishCounting/            # CountingUtils, OpenCVWrapper
│   ├── FrameSources/            # Camera, Album, UVC
│   └── Visualization/           # YOLOView
├── Packages/                    # Backend dependencies
│   └── opencv2/
│       ├── ios/opencv2.framework/
│       ├── ios-simulator/opencv2.framework/
│       ├── OpenCVBridge.h       # Obj-C++ bridge (added to Xcode project)
│       └── OpenCVBridge.mm      # Obj-C++ bridge (added to Xcode project)
├── YOLOiOSApp/                  # Example frontend implementation
│   ├── ViewController.swift
│   ├── Main.storyboard
│   └── FishCountModels/         # CoreML models
└── Package.swift
```

**Flow**: `FrameSource → AizipFishCount → TrackingDetector → ByteTracker → Counting → YOLOView`

## Development Rules

### Before Changes
1. Read Python reference (`../vision_yolo/`)
2. Explain planned changes
3. Get confirmation

### After Changes
1. **Always build**: Run build command below
2. Fix all errors before committing
3. Update docs if needed

### Code Quality
- Follow Apple Swift conventions
- Use `// MARK: - Section` for organization
- Add docstrings (Apple markup) for public APIs
- Use Accelerate framework for math (not manual loops)

## Build Command

**iPad Pro M4 Simulator**:
- XCode version: 26.2, iOS version: 26.2.
```bash
cd /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp && xcodebuild -configuration Debug -scheme YOLOiOSApp -destination 'platform=iOS Simulator,id=FCB1DD0F-9CBE-4344-B3A4-164660B57BBD' FRAMEWORK_SEARCH_PATHS="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/Packages/opencv2/ios-simulator" HEADER_SEARCH_PATHS="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/Packages/opencv2" SWIFT_OBJC_BRIDGING_HEADER="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp/YOLOiOSApp-Bridging-Header.h" OTHER_LDFLAGS="-lc++ -ObjC" OTHER_CPLUSPLUSFLAGS="-Wno-documentation -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-quoted-include-in-framework-header" -quiet | grep -E "error:|warning:|BUILD|SUCCEEDED|FAILED" | head -50
```

**Note**: Framework and header search paths are configured for on-device build in project.pbxproj:
- `FRAMEWORK_SEARCH_PATHS = "$(PROJECT_DIR)/../Packages/opencv2/ios/"`  (device) or `ios-simulator/` (simulator)
- `HEADER_SEARCH_PATHS = "$(PROJECT_DIR)/../Packages/opencv2/ios/"`

## Python → Swift Translation

| Python | Swift |
|:-------|:------|
| `np.array` operations | Accelerate framework (`cblas_*`) |
| `dict` state tracking | `[Int: Type]` with periodic cleanup |
| `cv2.*` operations | `OpenCVWrapper` (Objective-C++ bridge) |
| Hungarian matching | `MatchingUtils.swift` |

## Performance Targets

- **Frame processing**: <100ms (10+ FPS)
- **Inference**: <50ms (Neural Engine)
- **Memory**: Stable during 1-hour sessions
- **UI updates**: Main thread, 60 FPS

## Memory Management

- Clean up track state at certain frame intervals, e.g. every 30 frames
- Use weak references for delegates
- Release CVPixelBuffers properly
- Monitor with Xcode Instruments