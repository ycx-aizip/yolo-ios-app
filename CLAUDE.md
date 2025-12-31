# iOS Deployment Rules

## Structure

```
yolo-ios-app/
├── Sources/                            # Canonical source (edit here)
│   ├── AizipFishCount/                 # Backend
│   └── Visualization/                  # Frontend UI
│
├── AizipFishCountApp/                  # Dev project (Swift Package)
│   └── AizipFishCountApp.xcodeproj
│
├── AizipFishCountApp-Release/          # Release project (Framework)
│   ├── AizipFishCountApp.xcodeproj
│   └── AizipFishCount/                 # Copied from Sources/
│
├── Packages/opencv2/                   # OpenCV xcframework
├── build/                              # XCFramework output
└── Scripts/                            # Build scripts
```

**See**: `setup.md` for quick reference

**Flow**: `Camera → session.processFrame() → TrackingDetector → ByteTracker → Delegate → UI`

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

## Build Commands

### Development Build (Daily Work)

**Use**: `AizipFishCountApp` project with Swift Package

```bash
cd /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/AizipFishCountApp && \
xcodebuild -configuration Debug \
           -scheme AizipFishCountApp \
           -destination 'platform=iOS Simulator,id=FCB1DD0F-9CBE-4344-B3A4-164660B57BBD' \
           HEADER_SEARCH_PATHS="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/Packages/opencv2" \
           SWIFT_OBJC_BRIDGING_HEADER="/Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app/AizipFishCountApp/AizipFishCountApp-Bridging-Header.h" \
           OTHER_LDFLAGS="-lc++ -ObjC" \
           OTHER_CPLUSPLUSFLAGS="-Wno-documentation -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-quoted-include-in-framework-header" \
           -quiet | grep -E "error:|warning:|BUILD|SUCCEEDED|FAILED" | head -50
```

**Xcode**: Open `AizipFishCountApp/AizipFishCountApp.xcodeproj`, select iPad Pro M4 simulator

### Release Build (Partner Distribution)

**Use**: `AizipFishCountApp-Release` project with Framework target

```bash
cd /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app
./Scripts/build_aizipfishcount_xcframework.sh
```

**Output**: `AizipFishCountApp-Release/Library/AizipFishCount.xcframework` (5.7MB)

**Protected**: ByteTracker, OCSort, TrackingDetector, counting logic (compiled binary)
**Exposed**: Public APIs only (FishCountingSession protocol, configuration types)

### OpenCV Integration

- Using `opencv2.xcframework` (universal binary, embedded with "Embed & Sign")
- XCFramework automatically selects correct architecture (device vs simulator)
- `HEADER_SEARCH_PATHS` required for OpenCVBridge.h (Obj-C++ bridging header)
- Framework search paths use `$(inherited)` (configured in project.pbxproj)
- **Note**: OpenCV calls are commented out in AizipFishCount framework

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