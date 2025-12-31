# Fish Counting SDK for iOS

**Version**: 1.2.1
**Developer**: Aizip Inc. & SoftBank
**Last Updated**: 2025-12-31

A specialized iOS SDK for real-time fish counting using computer vision and multi-object tracking. Provides binary framework for IP protection with customizable UI source code.

---

## Environment

### Requirements

- **Development**: macOS with Xcode 16.0+ (tested with Xcode 16.2)
- **Target Platform**: iOS 16.0+ (tested on iOS 18+, iPadOS 18+)
- **Hardware**: iPhone 14+ or iPad Pro recommended
- **Apple Developer Account**: Free account sufficient for development

### Package Contents

```
AizipFishCount/
├── AizipFishCountApp-Release/          # Example Xcode project
│   ├── AizipFishCountApp.xcodeproj     # Integration reference
│   ├── Library/
│   │   └── AizipFishCount.xcframework  # Binary SDK (5.7MB)
│   └── AizipFishCountApp/              # Example app
│       ├── ViewController.swift
│       ├── Main.storyboard
│       └── FishCountModels/            # CoreML models
│
├── Sources/
│   └── Visualization/                  # UI source code (customizable)
│       ├── UI.swift                    # FishCountView
│       ├── FishCountView+BackendDependencies.swift
│       └── YOLOCamera.swift
│
└── Packages/
    └── opencv2/                        # OpenCV dependency (optional)
        ├── opencv2.xcframework
        ├── OpenCVBridge.h
        └── OpenCVBridge.mm
```

### What's Included

**Binary Framework**:
- `AizipFishCount.xcframework` - Compiled tracking/counting algorithms
  - Aizip Fish Counting models and implementations.
  - Public API only (no source code)

**UI Source Code** (Customizable):
- `Sources/Visualization/` - Complete UI implementation
  - Modify colors, layouts, controls
  - Add custom feature
  - Full source access

**Example Project**:
- Working integration reference
- Storyboard setup
- Build configuration

---

## Implementation

### Integration Steps

#### 1. Add Binary Framework

Open your Xcode project:
```
File → Add Files to "YourProject"
→ Select AizipFishCountApp-Release/Library/AizipFishCount.xcframework
→ Ensure "Copy items if needed" is checked
```

Configure target:
```
Target → General → Frameworks, Libraries, and Embedded Content
→ Set AizipFishCount.xcframework to "Embed & Sign"
```

#### 2. Add UI Source Files

Add Visualization source:
```
File → Add Files to "YourProject"
→ Select Sources/Visualization/ folder
→ Ensure "Create groups" is selected
→ Add to your app target
```

#### 3. Configure Build Settings

If using OpenCV (optional):
```
Target → Build Settings
→ Search "Header Search Paths"
→ Add: "$(PROJECT_DIR)/../Packages/opencv2"

→ Search "Bridging Header"
→ Set: "YourApp-Bridging-Header.h"

→ Search "Other Linker Flags"
→ Add: -lc++ -ObjC
```

#### 4. Update Storyboard (if using Interface Builder)

If using storyboard, update FishCountView module:
```xml
<view customClass="FishCountView"
      customModule="YourAppName"
      customModuleProvider="target">
```

**Important**: Set `customModule` to your app name, not "Visualization"

#### 5. Import and Use

```swift
import AizipFishCount

class ViewController: UIViewController, FishCountingSessionDelegate {
    @IBOutlet weak var fishCountView: FishCountView!

    override func viewDidLoad() {
        super.viewDidLoad()

        // Configure counting
        fishCountView.actionDelegate = self
        fishCountView.countingDirection = .bottomToTop
        fishCountView.countingThresholds = [0.3, 0.7]

        // Load model
        let modelURL = Bundle.main.url(forResource: "FishCount_v12",
                                       withExtension: "mlmodelc")!
        fishCountView.startSession(modelURL: modelURL)
    }

    // Handle counting results
    func onCountingResultUpdated(_ result: CountingResult) {
        print("Total count: \(result.totalCount)")
        print("Active tracks: \(result.tracks.count)")
    }

    func onCalibrationCompleted(_ summary: CalibrationSummary) {
        print("Calibration: \(summary.direction)")
    }
}
```

### Public API Reference

**Session Protocol**:
```swift
public protocol FishCountingSession {
    var delegate: FishCountingSessionDelegate? { get set }

    func startSession(modelURL: URL,
                     countingDirection: CountingDirection,
                     calibrationFrameCount: Int)
    func stopSession()
    func processFrame(_ pixelBuffer: CVPixelBuffer)
    func updateThresholds(_ threshold1: CGFloat, _ threshold2: CGFloat)
    func resetCounting()
}
```

**Data Types**:
```swift
public struct CountingResult {
    public let frameCount: Int
    public let totalCount: Int
    public let tracks: [STrack]
    public let detections: [YOLOResult]
}

public enum CountingDirection {
    case topToBottom, bottomToTop, leftToRight, rightToRight
}

public struct CalibrationSummary {
    public let direction: CountingDirection
    public let threshold1: CGFloat
    public let threshold2: CGFloat
}
```

### Customizing UI

Modify `Sources/Visualization/UI.swift` to customize:
- **Colors**: Change track colors, threshold line colors
- **Layout**: Adjust control positions, sizes
- **Features**: Add custom overlays, statistics displays
- **Behavior**: Modify gesture handling, interactions

Example - Change threshold colors:
```swift
// In UI.swift, modify drawThresholdLines()
context.setStrokeColor(UIColor.red.cgColor)      // First threshold
context.setStrokeColor(UIColor.yellow.cgColor)   // Second threshold
```

---

## Usage

### Basic Fish Counting

**Step 1: Launch App**
- Opens with fish counting mode enabled
- Camera permissions required

**Step 2: Configure Thresholds**
- Red line: First counting threshold
- Yellow line: Second counting threshold
- Drag sliders to adjust threshold positions

**Step 3: Set Direction**
- Default: Bottom-to-Top
- Change via direction button for different orientations

**Step 4: Monitor Counting**
- **Dark Blue boxes**: Newly detected fish
- **Light Blue boxes**: Tracked fish
- **Green boxes**: Successfully counted fish
- Count increments when fish cross threshold in correct direction

**Step 5: Reset**
- Tap RESET button to clear count and restart

### Advanced Configuration

**Manual Threshold Setup**:
```swift
fishCountView.countingThresholds = [0.25, 0.75]  // [threshold1, threshold2]
fishCountView.countingDirection = .bottomToTop
```

**Auto-Calibration** (Phase 2 only):
```swift
// Auto-detect counting direction from fish movement
// Note: Phase 1 (OpenCV threshold detection) is disabled
fishCountView.startSession(modelURL: modelURL)
// Calibration runs for first 30 frames
// Direction auto-detected and reported via delegate
```

**Video Sources**:
- **Camera**: Live camera feed (default)
- **Album**: Video file playback
- **UVC**: External USB camera (iPad only)

### Tracking Parameters

Default configuration (optimal for most scenarios):
```swift
// Detection
confidenceThreshold: 0.25    // Minimum detection confidence
iouThreshold: 0.45           // Non-maximum suppression

// Tracking
trackThreshold: 0.6          // Track creation threshold
matchThreshold: 0.7          // Track-detection matching
maxAge: 30                   // Frames before track deletion

// Counting
countingThresholds: [0.3, 0.7]  // Zone boundaries (normalized)
countingDirection: .bottomToTop  // Fish movement direction
```

### Performance

**Typical Performance** (iPad Pro M4):
- **Inference**: 30-40 FPS
- **Tracking**: <10ms per frame
- **Memory**: <200MB stable
- **Accuracy**: 95%+ on clear water videos

**Optimization Tips**:
- Use recommended iOS 18+ for best Neural Engine performance
- Reduce video resolution for lower-end devices
- Adjust `confidenceThreshold` to filter false detections

### Troubleshooting

**App hangs at launch**:
- Verify storyboard `customModule` matches your app name
- Check `FishCountView` outlet is connected

**No detections appearing**:
- Lower `confidenceThreshold` (try 0.2)
- Verify model file is included in app bundle
- Check camera permissions granted

**Count not incrementing**:
- Verify thresholds are positioned in fish path
- Check `countingDirection` matches fish movement
- Ensure fish crosses both thresholds completely

**Build errors**:
- Ensure xcframework is set to "Embed & Sign"
- Verify Visualization source files added to target
- Check build settings if using OpenCV

### Example Project

See `AizipFishCountApp-Release/AizipFishCountApp.xcodeproj` for:
- Complete storyboard integration
- Model loading and management
- UI customization examples
- Delegate implementation

Build and run to test the SDK before integration.

---

## Technical Support

**Email**: yenchi@aizip.ai, yuchen@aizip.ai
**Documentation**: See example project for integration reference
**Issues**: Contact support with build logs and error descriptions
