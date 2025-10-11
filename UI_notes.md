# UI Development Guide for iOS Fish Counting App

This guide provides Softbank collaborators with clear instructions for safely modifying the fish counting app interface.

## UI Files Organization

**Safe to Edit:**
- **`YOLOiOSApp/Main.storyboard`** - Interface Builder layout
- **`YOLOiOSApp/LaunchScreen.storyboard`** - App launch screen
- **`YOLOiOSApp/Assets.xcassets/`** - App icons and visual assets
- **`Sources/YOLO/Visualization/YOLOView.swift`** - Fish counting UI *(see editing guide below)*

**Do NOT Edit:**
- **`YOLOiOSApp/ViewController.swift`** - App startup logic
- **`YOLOiOSApp/Info.plist`** - App configuration (orientation locked to landscape right)
- **`YOLOiOSApp/AppDelegate.swift`** - App lifecycle management
- **`Sources/YOLO/Visualization/BoundingBoxView.swift`** - Detection visualization
- **`Sources/YOLO/Predictors/`** - AI detection logic
- **`Sources/YOLO/Tracking/`** - Object tracking algorithms
- **`Sources/YOLO/FishCounting/`** - Counting algorithms

## YOLOView.swift Editing Guide

`YOLOView.swift` (3518 lines) contains both UI and AI logic. **Only modify visual styling in designated sections.**

### SAFE TO EDIT

| Section | Line Range | What to Customize |
|---------|-----------|-------------------|
| **UI Setup** | ~898-1099 | Colors, fonts, sizes, button styling, fish count display format |
| **Layout Management** | ~1107-1562 | Positioning, frame calculations, spacing, margins, device-specific layouts |

**Main Customization Areas:**
- **Colors**: Line ~970-995 (threshold labels), ~1010-1032 (fish count display), ~1038-1042 (reset button)
- **Fonts**: Line ~899-948 (label fonts), ~1017-1023 (fish count font size)
- **Button Styling**: Line ~1009-1042 (auto-calibration, reset), ~1052-1096 (toolbar buttons)
- **Layout**: Line ~1115-1532 (landscape/portrait positioning)

### EDIT WITH CAUTION

| MARK Section | What You Can Edit | What to Avoid |
|-------------|------------------|---------------|
| **`// MARK: - UI Control Properties`** | Add new UI properties<br>Change access levels | Remove existing properties |
| **`// MARK: - Fish Counting Properties`** | Visual threshold properties<br>UI element declarations | Logic properties (`isCalibrating`, `countingDirection`) |
| **`// MARK: - UI Display Properties`** | All visual elements<br>Add new display elements | Activity indicator logic |
| **`// MARK: - UI Setup`**<br>**(Primary customization area)** | All colors, fonts, sizes<br>Button styling<br>Fish count display<br>Toolbar configuration | Target-action connections<br>Default config values<br>Gesture setup |
| **`// MARK: - Layout Management`**<br>**(Primary layout area)** | All positioning<br>Frame calculations<br>Device-specific layouts<br>Spacing and margins | `setupOverlayLayer()` calls<br>`setupThresholdLayers()` calls |
| **`// MARK: - User Interaction Handlers`** | Haptic feedback<br>Button state changes<br>Visual feedback | Slider value processing<br>Detector configuration<br>Model parameters |
| **`// MARK: - Video Controls`** | Button state management<br>Visual feedback | Frame source calls<br>Inference flags |
| **`// MARK: - Fish Counting Features`** | **CAUTION**: Visual styling only<br>Threshold line colors, width<br>Button appearance | Threshold calculations<br>Calibration workflow<br>Tracking interactions |

### DO NOT EDIT These Sections
- `// MARK: - VideoCaptureDelegate Implementation` - AI result processing
- `// MARK: - Model Management` - AI model handling
- `// MARK: - Detection Rendering` - AI result visualization
- `// MARK: - Frame Source Management` - Camera/video control
- All GoPro, UVC, and frame source switching sections