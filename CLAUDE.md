# iOS Deployment Rules

## Structure

```
yolo-ios-app/
├── Sources/YOLO/         # Swift Package
│   ├── Predictors/       # TrackingDetector
│   ├── Tracking/         # ByteTracker, STrack, KalmanFilter
│   ├── FishCounting/     # CountingUtils, OpenCVWrapper
│   ├── FrameSources/     # Camera, Album, GoPro, UVC
│   └── Visualization/    # YOLOView
└── YOLOiOSApp/           # Main app + models
```

**Flow**: `FrameSource → YOLO → TrackingDetector → ByteTracker → Counting → YOLOView`

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
```bash
cd /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp && xcodebuild -configuration Debug -scheme YOLOiOSApp -destination 'platform=iOS Simulator,id=496AC0E9-62B9-4335-B558-AD31BDD5F8D1' FRAMEWORK_SEARCH_PATHS="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp/Packages/ios-simulator /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp/Packages/MobileVLCKit.xcframework" SWIFT_OBJC_BRIDGING_HEADER="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp/YOLOiOSApp-Bridging-Header.h" OTHER_LDFLAGS="-lc++ -ObjC" OTHER_CPLUSPLUSFLAGS="-Wno-documentation -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-quoted-include-in-framework-header" -quiet | grep -E "error:|warning:|BUILD|SUCCEEDED|FAILED" | head -50
```

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

- Clean up track state every 30 frames
- Use weak references for delegates
- Release CVPixelBuffers properly
- Monitor with Xcode Instruments