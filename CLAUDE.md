# iOS Deployment Rules

## Structure

```
yolo-ios-app/                          # Development repo (partners get same structure)
├── Sources/                           # Swift Package (3 targets)
│   ├── AizipFishCount/                # Target 1: Backend → xcframework (binary for partners)
│   │   ├── PublicAPI/                 # Public session protocol, types
│   │   ├── Predictors/                # TrackingDetector (internal)
│   │   ├── Tracking/                  # ByteTracker, OCSort (internal)
│   │   └── FishCounting/              # Counting, OpenCV (internal)
│   ├── FrameSources/                  # Target 2: Frontend (source code for partners)
│   │   ├── FrameSource.swift          # Protocol
│   │   ├── CameraVideoSource.swift    # Camera capture
│   │   ├── AlbumVideoSource.swift     # Video playback
│   │   └── UVCVideoSource.swift       # External USB camera
│   └── Visualization/                 # Target 3: Frontend (source code for partners)
│       ├── YOLOView.swift             # Complete UI (2770 lines)
│       └── BoundingBoxView.swift      # Rendering
├── Packages/opencv2/                  # Backend dependency (embedded in xcframework)
│   ├── opencv2.xcframework/
│   ├── OpenCVBridge.h
│   └── OpenCVBridge.mm
├── YOLOiOSApp/YOLOiOSApp/             # iOS app example
│   ├── AppDelegate.swift              # App lifecycle
│   ├── ViewController.swift           # App UI integration
│   ├── ModelDownloadManager.swift     # App-specific
│   └── FishCountModels/               # CoreML models
└── Package.swift                      # Defines 3 targets
```

**Partner Distribution**: Same structure, but `AizipFishCount/` → `AizipFishCount.xcframework/` (binary)

**Flow**: `CameraVideoSource → session.processFrame() → TrackingDetector → ByteTracker → Delegate → YOLOView`

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
- Xcode version: 16.2, iOS version: 18.2
```bash
cd /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp && \
xcodebuild -configuration Debug \
           -scheme YOLOiOSApp \
           -destination 'platform=iOS Simulator,id=FCB1DD0F-9CBE-4344-B3A4-164660B57BBD' \
           HEADER_SEARCH_PATHS="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/Packages/opencv2" \
           SWIFT_OBJC_BRIDGING_HEADER="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/YOLOiOSApp/YOLOiOSApp-Bridging-Header.h" \
           OTHER_LDFLAGS="-lc++ -ObjC" \
           OTHER_CPLUSPLUSFLAGS="-Wno-documentation -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-quoted-include-in-framework-header" \
           -quiet | grep -E "error:|warning:|BUILD|SUCCEEDED|FAILED" | head -50
```

**OpenCV Framework**:
- Using `opencv2.xcframework` (universal binary, embedded with "Embed & Sign")
- XCFramework automatically selects correct architecture (device vs simulator)
- `HEADER_SEARCH_PATHS` required for OpenCVBridge.h (Obj-C++ bridging header)
- Framework search paths use `$(inherited)` (configured in project.pbxproj)

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