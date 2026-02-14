# iOS Development

Swift implementation of fish counting SDK with binary distribution. Development repository containing source code, build scripts, and example app.

## Structure

```
yolo-ios-app/
├── Sources/                        # Canonical source (edit here)
│   ├── AizipFishCount/             # Backend → compiled to xcframework
│   │   ├── PublicAPI/              # FishCountSession protocol
│   │   ├── Predictors/             # TrackingDetector
│   │   ├── Tracking/               # ByteTracker, OCSort
│   │   ├── FishCounting/           # ThresholdCounter
│   │   ├── FrameSources/           # Camera, Album, UVC
│   │   └── Utilities/              # Helpers
│   └── Visualization/              # Frontend → distributed as source
│       ├── UI.swift                # FishCountView
│       └── FishCountView+BackendDependencies.swift
├── AizipFishCountApp/              # Dev app (Swift Package Manager)
│   └── AizipFishCountApp.xcodeproj
├── Packages/opencv2/               # OpenCV dependency
├── Scripts/
│   ├── build_aizipfishcount_xcframework.sh  # Build binary
│   └── release.sh                           # Package for distribution
└── .build/                         # Build output
```

## Development Rules

### Before Changes
1. Read Python reference (`../vision_yolo/projects/fish_count/`)
2. Check implementation logs (`../docs/Implementations/`)
3. Explain planned changes

### During Development
- **Code Style**: Swift conventions, `// MARK: -` for sections
- **Documentation**: Docstrings for public APIs
- **Performance**: Use Accelerate framework for math operations
- **Testing**: Test with all frame sources (Camera, Album, UVC)

### After Changes
1. Build: `open AizipFishCountApp/AizipFishCountApp.xcodeproj`
2. Test on iPad Pro M4 simulator (or device)
3. Fix all build errors and warnings
4. Update `../docs/Implementations/` if architecture changed

### Build Commands

**Development (daily)**:
```bash
# Xcode with Swift Package Manager (fast incremental builds) set up already in the XCode Project before using this command.
```bash
cd AizipFishCountApp
xcodebuild -configuration Debug \
           -scheme AizipFishCountApp \
           -destination 'platform=iOS Simulator,id=FCB1DD0F-9CBE-4344-B3A4-164660B57BBD' \
           HEADER_SEARCH_PATHS="../Packages/opencv2" \
           SWIFT_OBJC_BRIDGING_HEADER="AizipFishCountApp/AizipFishCountApp-Bridging-Header.h" \
           OTHER_LDFLAGS="-lc++ -ObjC"
```

**Release (distribution)**:
```bash
./Scripts/build_aizipfishcount_xcframework.sh  # Build xcframework
./Scripts/release.sh                            # Package to ../Aizip_softbank_fishcount_ipad/
```

### Performance Targets
- Frame processing: 30+ FPS
- Inference: <50ms (Neural Engine)
- Memory: Stable during 1-hour sessions

## Document Reference

- **Architecture**: `../docs/architecture.md` - System design
- **Implementations**: `../docs/Implementations/*.md` - Implementation details of the core modules
- **Preview**: `../docs/preview/*.md` - Review of previous implemented modules or algorithms
- **Python Reference**: `../vision_yolo/CLAUDE.md` - Original algorithms
