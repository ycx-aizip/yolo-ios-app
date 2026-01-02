# iOS Development Rules

## Structure

```
/
├── Sources/                            # Canonical source (edit here)
│   ├── AizipFishCount/                 # Backend → xcframework
│   └── Visualization/                  # Frontend → source
├── AizipFishCountApp/                  # Dev (Swift Package)
├── Packages/opencv2/                   # Packages
├── Scripts/                            # Build and Release scripts
└── .build/                             # XCode and XCFramework output
```

**Release repo**: `../Aizip_softbank_fishcount_ipad/` (separate Git repo)

---

## Development Rules

### Before Changes
1. Read Python reference (`../vision_yolo/`)
2. Explain planned changes

### After Changes
1. Build and test
2. Fix all errors
3. Update docs if needed

### Code Quality
- Apple Swift conventions
- `// MARK: - Section` for organization
- Docstrings for public APIs
- Accelerate framework for math

---

## Build Commands

### Development (Daily)

```bash
cd AizipFishCountApp
xcodebuild -configuration Debug \
           -scheme AizipFishCountApp \
           -destination 'platform=iOS Simulator,id=FCB1DD0F-9CBE-4344-B3A4-164660B57BBD' \
           HEADER_SEARCH_PATHS="../Packages/opencv2" \
           SWIFT_OBJC_BRIDGING_HEADER="AizipFishCountApp/AizipFishCountApp-Bridging-Header.h" \
           OTHER_LDFLAGS="-lc++ -ObjC"
```

**Xcode**: `open AizipFishCountApp/AizipFishCountApp.xcodeproj` (iPad Pro M4 simulator)

### Release (Distribution)

**Build xcframework**:
```bash
./Scripts/build_aizipfishcount_xcframework.sh
```
Output: `.build/AizipFishCount.xcframework` (5.7MB)

**Package for partners**:
```bash
./Scripts/release.sh
```
Output: `../Aizip_softbank_fishcount_ipad/`

---

## Python → Swift Translation

| Python | Swift |
|:-------|:------|
| `np.array` operations | Accelerate (`cblas_*`, `vDSP_*`) |
| `dict` state tracking | `[Int: Type]` with cleanup |
| `cv2.*` operations | Disabled (was `OpenCVWrapper`) |
| Hungarian matching | `MatchingUtils.swift` |

---

## Performance Targets

- **Frame processing**: <100ms (10+ FPS)
- **Inference**: <50ms (Neural Engine)
- **Memory**: Stable during 1-hour sessions

---

## Memory Management

- Clean track state every 30 frames
- Weak references for delegates
- Release CVPixelBuffers properly
