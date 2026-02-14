# Fish Counting iOS SDK - Development Repository

**Version**: 1.3.2
**Developer**: Aizip Inc.

Development repository for Fish Counting iOS SDK. Contains source code, build scripts, and example app.

---

## Architecture

**Backend** (`AizipFishCount`):
- **ByteTracker** - Multi-object tracking (SORT-based)
- **OCSort** - Kalman filter tracking with velocity direction consistency
- **ThresholdCounter** - Threshold-based counting state machine
- **TrackingDetector** - Coordinates detection → tracking → counting pipeline
- **PublicAPI** - Protocol-based interface for Visualization layer
- **Distribution**: Compiled to `AizipFishCount.xcframework` (binary, IP protected)

**Frontend** (`Visualization`):
- **FishCountView** - Main UI component with drawing, controls, gestures
- **YOLOCamera** - Camera integration and frame capture
- **FishCountView+BackendDependencies** - Only file that imports AizipFishCount
- **Distribution**: Source code (partners can customize)

**Example iOS App XCode Project** (`AizipFishCountApp`):
- Development reference using Swift Package Manager (SPM)
- SPM imports `Sources/AizipFishCount/` and `Sources/Visualization/` directly
- Storyboard integration, model loading, delegate implementation
- Camera/Album/UVC video sources
- Fast incremental builds during development

**Separation Strategy**:
- Visualization uses `FishCountingSession` protocol (never concrete `TrackingDetector`)
- Backend dependencies isolated to single file
- PublicAPI provides clean abstraction layer

---

## Sources

### `Sources/AizipFishCount/` (Backend - becomes xcframework)

**PublicAPI/**:
- `FishCountSession.swift` - Main protocol
- `SessionAdapter.swift` - Wraps TrackingDetector
- `PublicTypes.swift` - CountingResult, CountingDirection, CalibrationSummary, etc.
- `PublicDelegate.swift` - FishCountingSessionDelegate

**Predictors/**:
- `TrackingDetector.swift` - Main coordinator: detection → tracking → counting
- `ObjectDetector.swift` - YOLO model wrapper
- `BasePredictor.swift`, `Predictor.swift` - Model execution framework

**Tracking/**:
- `ByteTracker.swift` - SORT-based multi-object tracking
- `OCSort.swift` - Kalman filter tracking with observation history
- `STrack.swift` - Track state management
- `KalmanFilter.swift`, `OCKalmanFilter.swift` - Motion prediction
- `MatchingUtils.swift` - Hungarian algorithm, IoU variants (GIoU, DIoU, CIoU)

**FishCounting/**:
- `ThresholdCounter.swift` - Counting logic (2-threshold, direction-aware)
- `CountingUtils.swift` - Configuration, calibration (OpenCV calls disabled)
- `Movement.swift` - Movement analysis for Phase 2 calibration

**FrameSources/**:
- `FrameSource.swift` - Abstract interface
- `CameraVideoSource.swift` - AVFoundation camera
- `AlbumVideoSource.swift` - Video file playback
- `UVCVideoSource.swift` - External USB camera (iPad only)

**Utilities/**:
- `NonMaxSuppression.swift` - Post-processing
- `UnifiedCoordinateSystem.swift` - Coordinate transformations
- `YOLOResult.swift`, `YOLOTask.swift` - Data structures

### `Sources/Visualization/` (Frontend - source distribution)

**UI.swift**:
- `FishCountView` - Main view with drawing (bounding boxes, threshold lines, labels)
- Control UI (sliders, buttons, direction selector)
- Gesture handling (pinch, tap)
- Drawing logic for tracks, detections, overlays

**FishCountView+BackendDependencies.swift**:
- Only file importing `AizipFishCount`
- Creates `SessionAdapter`, handles delegate callbacks
- Isolates backend coupling

**YOLOCamera.swift**:
- SwiftUI wrapper for FishCountView
- Camera permission handling

---

## Packages

### `Packages/opencv2/`

**Contents**:
- `opencv2.xcframework` - OpenCV binary (device + simulator)
- `OpenCVBridge.h` - Objective-C header
- `OpenCVBridge.mm` - Objective-C++ implementation

**Status**:
- **NOT included in AizipFishCount.xcframework**
- Bridging headers incompatible with `BUILD_LIBRARY_FOR_DISTRIBUTION=YES`
- All OpenCV calls commented out in framework source
- Phase 1 calibration (threshold detection) disabled
- Linked at app level (not framework level)

**Partners**:
- Can optionally use opencv2 in their app
- Not required for fish counting to work
- Only needed if re-enabling Phase 1 calibration

---

## Scripts

### `build_aizipfishcount_xcframework.sh`

**Purpose**: Build binary framework for distribution

**Process**:
1. Syncs `Sources/AizipFishCount/` → `../Aizip_softbank_fishcount_ipad/AizipFishCountApp/AizipFishCount/`
2. Archives for iOS device (arm64)
3. Archives for iOS simulator (arm64 + x86_64)
4. Creates xcframework: `.build/AizipFishCount.xcframework` (5.7MB)

**Output**: Binary framework with public API only (`.swiftinterface`)

**Usage**: `./Scripts/build_aizipfishcount_xcframework.sh`

### `release.sh`

**Purpose**: Package complete distribution for partners

**Process**:
1. Verifies xcframework exists in `.build/`
2. Copies xcframework → `../Aizip_softbank_fishcount_ipad/Packages/`
3. Copies Visualization source → `../Aizip_softbank_fishcount_ipad/Sources/`
4. Copies CoreML models → `../Aizip_softbank_fishcount_ipad/AizipFishCountApp/FishCountModels/`
5. **Removes** `AizipFishCount/` framework source (build artifact, not for distribution)
6. **Does NOT copy** opencv2 (partners add if needed)

**Output**: Complete SDK in `Aizip_softbank_fishcount_ipad/`

**Usage**: `./Scripts/release.sh`

---

## Usage

```
Development (this repo):
  Sources/AizipFishCount/ → Edit backend code
  Sources/Visualization/ → Edit UI code
  AizipFishCountApp/ → Test with Swift Package Manager (SPM imports source directly)

Release Build:
  build_aizipfishcount_xcframework.sh → .build/AizipFishCount.xcframework (binary)
  release.sh → Package to ../Aizip_softbank_fishcount_ipad/

Partner Distribution:
  Partners receive: Aizip_softbank_fishcount_ipad/ repository
  - AizipFishCount.xcframework (binary, IP protected)
  - Sources/Visualization/ (source code, customizable)
  - Example app uses xcframework + Visualization source
```

---

## Documentation
- `../docs/Implementations/ios_release.md` - Project level documentations including preview, implementation notes on all development phases, models -> deployment.
