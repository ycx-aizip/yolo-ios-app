# Fish Counting App for iOS

A specialized iOS application developed by Aizip inc. and Softbank that uses computer vision and object tracking to count fish swimming through video frames in real-time.

## Project Architecture

![Fish Counting App Logic](fish_count_overview.svg)
*Figure. Overview of the fish countign iOS app*

### 1. AI/Vision Core (`yolo-ios-app/Sources/YOLO/`)

The YOLO Swift package is organized into functional subdirectories:

**Core AI Logic** (`Predictors/`, `Tracking/`, `FishCounting/`):
- `Predictors/TrackingDetector.swift` - Main fish counting coordinator
- `Tracking/ByteTracker.swift` - Multi-object tracking implementation  
- `Tracking/STrack.swift` - Individual track state management
- `Tracking/KalmanFilter.swift` - Motion prediction algorithms
- `Tracking/MatchingUtils.swift` - Track association and data association
- `FishCounting/ThresholdCounter.swift` - Threshold-based counting logic
- `FishCounting/CountingUtils.swift` - Counting configuration management
- `FishCounting/OpenCVWrapper.swift` - Computer vision utilities for calibration

**Model Infrastructure** (`Predictors/`):
- `ObjectDetector.swift`, `Classifier.swift`, `Segmenter.swift` - Detection model interfaces
- `BasePredictor.swift`, `Predictor.swift` - Model execution framework

**Video Processing** (`FrameSources/`):
- `FrameSource.swift` - Abstract video source interface
- `CameraVideoSource.swift` - Live camera capture
- `AlbumVideoSource.swift` - Video file playback
- `GoProSource.swift` - GoPro RTSP streaming
- `UVCVideoSource.swift` - External USB camera support

**Visualization** (`Visualization/`):
- `Visualization/YOLOView.swift` - Core fish counting UI component
- `Visualization/BoundingBoxView.swift` - Detection visualization
- `Visualization/Plot.swift` - Visualization utilities

**Utilities** (`Utilities/`):
- `Utilities/UnifiedCoordinateSystem.swift` - Coordinate transformations
- `Utilities/NonMaxSuppression.swift` - Post-processing algorithms
- `Utilities/YOLOResult.swift`, `YOLOTask.swift` - Data structures

### 2. UI Layer (`YOLOiOSApp/`, `Sources/YOLO/YOLOView.swift`, and `Sources/YOLO/BoundingBoxView.swift`)

**Main UI Files** (primary focus for external developers):
- `YOLOiOSApp/ViewController.swift` - Main app controller and model selection
- `YOLOiOSApp/Main.storyboard` - Interface Builder layout definitions
- `YOLOiOSApp/LaunchScreen.storyboard` - App launch screen
- `Sources/YOLO/YOLOView.swift` - Core fish counting UI component
- `Sources/YOLO/BoundingBoxView.swift` - Detection visualization

**Supporting UI Files**:
- `YOLOiOSApp/Assets.xcassets/` - App icons, images, and visual assets
- `YOLOiOSApp/ModelDownloadManager.swift` - Model download and management
- `YOLOiOSApp/RemoteModels.swift` - Remote model configuration

## Installation Guide

For testing, we normally provide *TestFlights*, for developers, see below.

### Requirements

- **Development Environment**: macOS with Xcode 16.0+ (tested with Xcode 16.2)
- **Target Platform**: iOS 17.0+ (tested with iOS 18+, iPadOS 18+)
- **Hardware**: iPhone 14+ or iPad if using UVC external camera.
- **Apple Developer Account**: Free account sufficient for development. To use GoPro, need developer account with `multi-casting network` entitlement.

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone [repository-url]
   cd softbank_fishcount_iphone14
   ```

2. **Download VLCKit (for GoPro, requirement may be removed in the future)**
   - [MobileVLCKit-3.6.1b1-8e652244-ac310b4b.tar.xz](https://artifacts.videolan.org/VLCKit/MobileVLCKit/MobileVLCKit-3.6.1b1-8e652244-ac310b4b.tar.xz).
   - Extract the package. Copy `MobileVLCKit.xcframe` folder to `YOLOiOSApp/Packages`

2. **Open in Xcode**:
   - Launch Xcode
   - Open `./YOLOiOSApp/YOLOiOSApp.xcodeproj`

3. **Configure Developer Account**:
   - Xcode → Preferences → Accounts
   - Add your Apple ID
   - Select development team in project settings

4. **Build and Deploy**:
   - Build using XCode.

5. **Device Installation**:
   - Connect iPad via USB
   - Build and run from Xcode
   - Trust developer certificate in iOS Settings → General → Device Management

## Usage Guide

### Basic Operation

1. **Launch App**: Fish counting mode loads automatically with optimized model
2. **Camera Setup**: Grant camera permissions, point at fish in clear water
3. **Threshold Configuration**: 
   - Red line: First counting threshold
   - Yellow line: Second counting threshold
   - Adjust via sliders for optimal detection zone
4. **Direction Settings**: Configure counting direction (Bottom-to-Top default)
5. **Monitoring**: Watch real-time count and color-coded fish tracking
   - *Dark Blue*: Newly detected fish (not yet tracked)
   - *Light Blue*: Established tracked fish (crossing thresholds)
   - *Green*: Successfully counted fish (crossed threshold in correct direction)
6. **Reset Functionality**: Clear count and restart session
7. **Manual Calibration**: Fine-tune thresholds and direction for specific conditions

### Advanced Features

- **Multiple Video Sources**: Camera, album videos, external cameras
- **Model Selection**: Switch between different fish detection models  
- **Auto-Calibration (Optional)**: Tap "AUTO" for intelligent threshold detection

### Technical Support

For development assistance or questions about the app logic:
- Email: [yenchi@aizip.ai] or [yuchen@aizip.ai]  