# iOS SDK Development — Context

## Overview

Swift implementation of the fish counting SDK with binary distribution. Development repository containing source code, build scripts, and an example app. Backend (`AizipFishCount`) compiles to an xcframework; frontend (`Visualization`) ships as source. Maintain algorithmic parity with the Python reference in `../vision_yolo/`.

## Structure

```
yolo-ios-app/
├── Sources/                        # Canonical source (edit here)
│   ├── AizipFishCount/             # Backend → compiled to xcframework
│   │   ├── PublicAPI/              # FishCountSession protocol
│   │   ├── Predictors/             # TrackingDetector, ObjectDetector
│   │   ├── Tracking/               # ByteTracker, OCSort, STrack, Kalman filters
│   │   ├── FishCounting/           # ThresholdCounter
│   │   ├── FrameSources/           # Camera, Album, UVC
│   │   └── Utilities/              # Helpers, coordinate transforms, NMS
│   └── Visualization/              # Frontend → distributed as source
│       ├── UI.swift                # FishCountView
│       └── FishCountView+BackendDependencies.swift
├── AizipFishCountApp/              # Dev app (Swift Package Manager)
│   └── AizipFishCountApp.xcodeproj
├── Packages/opencv2/               # OpenCV dependency (app-level only)
├── Scripts/
│   ├── build_aizipfishcount_xcframework.sh  # Build binary
│   └── release.sh                           # Package for distribution
└── .build/                         # Build output
```

## Rules

### Design principles
- Follow **SOLID**: single responsibility, intentional public/private surface, swappable implementations, small role-focused interfaces, core depends on abstractions.
- TDD: write tests first, then implement. Every test must validate real behavior — no trivial or placeholder tests. Test directory mirrors `Sources/` structure.
- Comments: only when the *why* isn't obvious from the code. No section dividers, no restating what the code does.

### Before changes
1. Read the Python reference (`../vision_yolo/projects/fish_count/`).
2. Check implementation logs (`../docs/Implementations/`).
3. Explain planned changes.

### During development
- **Code style**: Swift conventions, `// MARK: -` for sections.
- **Documentation**: docstrings for public APIs.
- **Performance**: use the Accelerate framework for math operations.
- **Debug logging**: toggle with `TrackingParameters.debugLogging` in `ThresholdCounter.swift`; guard new logs with `if TrackingParameters.debugLogging { ... }`.
- **Testing**: test with all frame sources (Camera, Album, UVC).

### After changes
1. Build the dev app (see below).
2. Test on iPad Pro simulator (or device).
3. Fix all build errors and warnings.
4. Update `../docs/Implementations/` if architecture changed.

### Build commands

**Development (daily)** — Xcode + Swift Package Manager (fast incremental builds):
```bash
cd /Users/xxb9075/Documents/softbank_fishcount_ios/yolo-ios-app/AizipFishCountApp

xcodebuild clean -scheme AizipFishCountApp   # clean if needed

xcodebuild -configuration Debug \
           -scheme AizipFishCountApp \
           -destination 'platform=iOS Simulator,name=iPad Pro 11-inch (M4)' \
           HEADER_SEARCH_PATHS="/Users/xxb9075/Documents/softbank_fishcount_ios/yolo-ios-app/Packages/opencv2/" \
           SWIFT_OBJC_BRIDGING_HEADER="/Users/xxb9075/Documents/softbank_fishcount_ios/yolo-ios-app/AizipFishCountApp/AizipFIshCountApp-Bridging-Header.h" \
           OTHER_LDFLAGS="-lc++ -ObjC" \
           OTHER_CPLUSPLUSFLAGS="-Wno-documentation -Wno-documentation-deprecated-sync -Wno-documentation-unknown-command -Wno-quoted-include-in-framework-header" \
           -quiet | grep -E "error:|warning:|BUILD|SUCCEEDED|FAILED" | head -50
```

**Release (distribution)**:
```bash
./Scripts/build_aizipfishcount_xcframework.sh  # Build xcframework
./Scripts/release.sh                            # Package to ../Aizip_softbank_fishcount_ipad/
```

### Performance targets
- Frame processing: 30+ FPS
- Inference: <50ms (Neural Engine)
- Memory: stable during 1-hour sessions

## Document reference

- **Architecture**: `../docs/architecture.md` — system design
- **Implementations**: `../docs/Implementations/*.md` — implementation details of the core modules
- **Preview**: `../docs/preview/*.md` — review of previously implemented modules or algorithms
- **Python reference**: `../vision_yolo/CLAUDE.md` — original algorithms
