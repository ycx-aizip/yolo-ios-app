# Project Setup

## Two Xcode Projects

### AizipFishCountApp/ (Development)
- **Use for**: Daily development
- **Backend**: Swift Package from `Sources/AizipFishCount/`
- **Frontend**: Swift Package from `Sources/Visualization/`
- **Dependencies**: `Packages/opencv2/`, `ZIPFoundation`
- **Build**: Fast incremental builds
- **Xcode**: Open `AizipFishCountApp/AizipFishCountApp.xcodeproj`

### AizipFishCountApp-Release/ (Distribution)
- **Use for**: Building XCFramework for partners
- **Backend**: Framework target with source from `AizipFishCountApp-Release/AizipFishCount/`
- **Frontend**: Source files from `Sources/Visualization/` (referenced, not copied)
- **XCFramework**: `AizipFishCountApp-Release/Library/AizipFishCount.xcframework`
- **Build**: Run `./Scripts/build_aizipfishcount_xcframework.sh`
- **Output**: 5.7MB binary (tracking/counting logic protected)

## Workflow

### Daily Development
1. Edit source code:
   - `Sources/AizipFishCount/` - Backend logic
   - `Sources/Visualization/` - UI components
2. Test in Dev project:
   - Open `AizipFishCountApp/AizipFishCountApp.xcodeproj`
   - Build and run on iPad Pro simulator
3. Commit changes to `Sources/`

### Creating Release for Partners
1. **Sync source code**:
   - Copy changes from `Sources/AizipFishCount/` to `AizipFishCountApp-Release/AizipFishCount/`
   - Or: re-add files as references in Xcode (one-time)

2. **Build XCFramework**:
   ```bash
   cd /Users/xxb9075/Documents/softbank_fishcount_iphone14/yolo-ios-app
   ./Scripts/build_aizipfishcount_xcframework.sh
   ```

3. **Test Release build**:
   - Open `AizipFishCountApp-Release/AizipFishCountApp.xcodeproj`
   - Verify XCFramework is linked (General → Frameworks)
   - Build and run to test

4. **Distribute**:
   - Package: `AizipFishCountApp-Release/Library/AizipFishCount.xcframework`
   - Include: `Sources/Visualization/` source files
   - Include: `Packages/opencv2/` (if needed by app)

## What's Protected (Binary)

The XCFramework contains compiled binary code for:
- `ByteTracker` - Multi-object tracking
- `OCSort` - Kalman filtering
- `TrackingDetector` - Detection + tracking integration
- `CountingLogic` - Fish counting algorithms
- Hungarian matching, IoU calculations

**Partners receive**:
- Binary XCFramework (no source code for backend)
- Visualization source code (UI components)
- Public API interfaces only

## What's Exposed (Source)

Partners get source code for:
- `Sources/Visualization/UI.swift` - FishCountView UI
- `Sources/Visualization/YOLOCamera.swift` - Camera wrapper
- Public protocols: `FishCountingSession`, `CountingResult`, etc.

## Code Signing

- **Development builds**: Unsigned XCFramework (faster builds)
- **Production distribution**: Partners sign during integration
- **Optional**: Remove `CODE_SIGNING_*` flags from build script to sign during dev

## Key Files

- **Sources/**: Canonical source code (edit here)
- **AizipFishCountApp/**: Dev project (Swift Package)
- **AizipFishCountApp-Release/**: Release project (Framework target)
- **AizipFishCountApp-Release/Library/**: XCFramework output location
- **Scripts/build_aizipfishcount_xcframework.sh**: Build automation
- **CLAUDE.md**: Development rules and build commands
