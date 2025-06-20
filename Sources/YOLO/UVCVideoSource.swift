// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, managing UVC external camera capture for real-time inference.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The UVCVideoSource component manages external UVC camera capture for real-time object detection
//  on iPad devices with USB-C ports running iPadOS 17+. It extends the camera capture pipeline
//  to support USB Video Class (UVC) compliant devices such as Logitech webcams, HDMI capture cards,
//  and other external cameras. The implementation leverages Apple's native UVC support introduced
//  in iPadOS 17 through AVFoundation's external camera discovery APIs.

import AVFoundation
import CoreVideo
import UIKit
import Vision
import Foundation

/// UVC Video Source for external camera support on iPad
/// 
/// Provides UVC (USB Video Class) camera support for iPad devices with USB-C ports.
/// Adapts the existing camera capture pipeline to work with external cameras discovered
/// through AVFoundation's external device APIs introduced in iPadOS 17.
///
/// **Key Features:**
/// - Automatic UVC device discovery and connection
/// - Adaptive resolution (HD 720p → 1080p → medium fallback)
/// - Real-time device connection monitoring
/// - Smooth orientation transitions with consistent video display
/// - Performance metrics and logging
@preconcurrency
class UVCVideoSource: NSObject, FrameSource, @unchecked Sendable {
    
    // MARK: - FrameSource Protocol Properties
    
    var predictor: FrameProcessor!
    var previewLayer: AVCaptureVideoPreviewLayer?
    weak var videoCaptureDelegate: VideoCaptureDelegate?
    weak var frameSourceDelegate: FrameSourceDelegate?
    var captureDevice: AVCaptureDevice?
    let captureSession = AVCaptureSession()
    var videoInput: AVCaptureDeviceInput?
    let videoOutput = AVCaptureVideoDataOutput()
    let cameraQueue = DispatchQueue(label: "uvc-camera-queue")
    var inferenceOK = true
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    
    /// The source type identifier
    var sourceType: FrameSourceType { return .uvc }
    
    /// The delegate to receive frames and performance metrics
    var delegate: FrameSourceDelegate? {
        get { return frameSourceDelegate }
        set { frameSourceDelegate = newValue }
    }
    
    // MARK: - Private Properties
    
    /// Tracks frame size capture state
    private var frameSizeCaptured = false
    
    /// Current pixel buffer for processing
    private var currentBuffer: CVPixelBuffer?
    
    /// Track the last known valid orientation for coordinate transformation
    private var lastKnownOrientation: UIDeviceOrientation = .portrait
    
    /// Prevents concurrent orientation updates
    private var isUpdatingOrientation = false
    
    /// Stores pending orientation update during concurrent access
    private var pendingOrientationUpdate: UIDeviceOrientation?
    
    /// Tracks if model is currently processing a frame
    private var isModelProcessing: Bool = false
    
    // MARK: - Performance Metrics
    
    /// Frame processing count for performance tracking
    private var frameProcessingCount = 0
    
    /// Timestamps for frame processing rate calculation
    private var frameProcessingTimestamps: [CFTimeInterval] = []
    
    /// Performance timing metrics
    private var lastFramePreparationTime: Double = 0
    private var lastConversionTime: Double = 0
    private var lastInferenceTime: Double = 0
    private var lastUITime: Double = 0
    private var lastTotalPipelineTime: Double = 0
    
    /// FPS calculation arrays
    private var frameTimestamps: [CFTimeInterval] = []
    private var pipelineStartTimes: [CFTimeInterval] = []
    private var pipelineCompleteTimes: [CFTimeInterval] = []
    
    // MARK: - UVC Configuration & Capabilities
    
    /// UVC camera capability information
    struct UVCCapabilities {
        let deviceName: String
        let modelID: String?
        let availableFormats: [(resolution: CGSize, frameRates: [Double], pixelFormat: String)]
        let zoomRange: (min: CGFloat, max: CGFloat)
        let supportedPresets: [AVCaptureSession.Preset]
        let activeFormat: (resolution: CGSize, frameRate: Double, pixelFormat: String)
    }
    
    /// UVC camera configuration with priority-based fallback
    struct UVCConfiguration {
        let targetResolution: CGSize
        let targetFrameRate: Double
        let targetZoomFactor: CGFloat
        let useWidestFOV: Bool
        let preferredPixelFormat: OSType?
        
        /// Default configuration: 1280×720 @ 60fps with 420f format
        /// This is the single place to modify UVC camera settings
        static let preferred = UVCConfiguration(
            targetResolution: CGSize(width: 1280, height: 720),
            targetFrameRate: 60.0,
            targetZoomFactor: 1.0,
            useWidestFOV: true,
            preferredPixelFormat: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
        )
    }
    
    /// Cached device capabilities and active configuration
    private var cachedCapabilities: UVCCapabilities?
    private var activeConfiguration: UVCConfiguration?
    
    // MARK: - Connection Monitoring
    
    /// Discovery session for monitoring device connections (iOS 17+)
    private var deviceDiscoverySession: AVCaptureDevice.DiscoverySession?
    
    /// Observer for device connection changes
    private var deviceConnectionObserver: NSKeyValueObservation?
    
    // MARK: - Initialization
    
    /// Initializes UVC video source with connection monitoring
    override init() {
        super.init()
        Task { @MainActor in
            self.setupConnectionMonitoring()
        }
    }
    
    deinit {
        cleanupConnectionMonitoring()
    }
    
    // MARK: - Connection Monitoring (Apple WWDC 2023 Recommendations)
    
    /// Sets up connection monitoring for external cameras as per Apple WWDC 2023 recommendations
    @MainActor
    private func setupConnectionMonitoring() {
        guard #available(iOS 17.0, *) else { return }
        
        deviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external],
            mediaType: .video,
            position: .unspecified
        )
        
        if let session = deviceDiscoverySession {
            deviceConnectionObserver = session.observe(\.devices, options: [.new]) { [weak self] _, _ in
                Task { @MainActor in
                    guard let self = self, let discoverySession = self.deviceDiscoverySession else { return }
                    self.handleDeviceListChange(session: discoverySession)
                }
            }
        }
        
        print("UVC: Connection monitoring setup complete")
    }
    
    /// Cleans up connection monitoring resources
    private func cleanupConnectionMonitoring() {
        deviceConnectionObserver?.invalidate()
        deviceConnectionObserver = nil
        deviceDiscoverySession = nil
        print("UVC: Connection monitoring cleanup complete")
    }
    
    /// Handles device list changes from the discovery session
    @MainActor
    private func handleDeviceListChange(session: AVCaptureDevice.DiscoverySession) {
        let currentDevices = session.devices
        print("UVC: Device list changed - \(currentDevices.count) external devices detected")
        
        guard let currentDevice = captureDevice else { return }
        
        let isStillConnected = currentDevices.contains { device in
            device.uniqueID == currentDevice.uniqueID && device.isConnected
        }
        
        if !isStillConnected {
            print("UVC: Current device \(currentDevice.localizedName) has been disconnected")
            handleDeviceDisconnection()
        }
    }
    
    /// Handles device disconnection gracefully
    @MainActor
    private func handleDeviceDisconnection() {
        print("UVC: Handling device disconnection")
        
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        
        captureDevice = nil
        
        // Notify delegates about disconnection if needed
        if let delegate = videoCaptureDelegate as? YOLOView {
            print("UVC: Device disconnected - consider switching back to camera source")
        }
    }

    // MARK: - Device Discovery & Management
    
    /// Applies device-specific optimizations for known UVC cameras
    private func applyDeviceSpecificWorkarounds(device: AVCaptureDevice) {
        let deviceName = device.localizedName.lowercased()
        
        if deviceName.contains("brio") || deviceName.contains("logitech") {
            print("UVC: \(device.localizedName) detected - premium UVC camera")
        } else if deviceName.contains("osmo") {
            print("UVC: \(device.localizedName) detected - action camera")
        } else {
            print("UVC: Generic UVC device detected: \(device.localizedName)")
        }
    }
    
    /// Discovers and returns the best available UVC camera device
    @MainActor
    static func bestUVCDevice() -> AVCaptureDevice? {
        guard #available(iOS 17.0, *) else {
            print("UVC: External cameras require iOS 17.0 or later")
            return nil
        }
        
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external],
            mediaType: .video,
            position: .unspecified
        )
        
        let validDevices = discoverySession.devices.filter { device in
            device.hasMediaType(.video) && device.isConnected && device.deviceType == .external
        }
        
        if let bestDevice = validDevices.first {
            print("UVC: Selected device: \(bestDevice.localizedName)")
            return bestDevice
        } else {
            print("UVC: No external cameras found")
            return nil
        }
    }
    
    // MARK: - Device Capability Analysis
    
    /// Analyzes UVC device capabilities for configuration optimization
    @MainActor
    private func analyzeUVCCapabilities(device: AVCaptureDevice) -> UVCCapabilities {
        var availableFormats: [(resolution: CGSize, frameRates: [Double], pixelFormat: String)] = []
        
        for format in device.formats {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let resolution = CGSize(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height))
            let frameRates = format.videoSupportedFrameRateRanges.map { Double($0.maxFrameRate) }.sorted()
            let pixelFormat = pixelFormatString(from: format.formatDescription)
            
            availableFormats.append((resolution: resolution, frameRates: frameRates, pixelFormat: pixelFormat))
        }
        
        availableFormats.sort { first, second in
            let firstArea = first.resolution.width * first.resolution.height
            let secondArea = second.resolution.width * second.resolution.height
            return firstArea > secondArea
        }
        
        let zoomRange = (min: device.minAvailableVideoZoomFactor, max: device.maxAvailableVideoZoomFactor)
        let supportedPresets: [AVCaptureSession.Preset] = [.high, .medium, .low, .hd1280x720, .hd1920x1080, .hd4K3840x2160, .photo].filter { device.supportsSessionPreset($0) }
        
        let currentFormat = device.activeFormat
        let dimensions = CMVideoFormatDescriptionGetDimensions(currentFormat.formatDescription)
        let resolution = CGSize(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height))
        let frameRate = Double(device.activeVideoMaxFrameDuration.timescale) / Double(device.activeVideoMaxFrameDuration.value)
        let pixelFormat = pixelFormatString(from: currentFormat.formatDescription)
        let activeFormat = (resolution: resolution, frameRate: frameRate, pixelFormat: pixelFormat)
        
        let capabilities = UVCCapabilities(
            deviceName: device.localizedName,
            modelID: device.modelID,
            availableFormats: availableFormats,
            zoomRange: zoomRange,
            supportedPresets: supportedPresets,
            activeFormat: activeFormat
        )
        
        logDeviceCapabilities(capabilities)
        return capabilities
    }
    
    /// Converts pixel format to human-readable string with format explanation
    /// 
    /// **Pixel Format Guide:**
    /// - **420v** (Video Range): YUV 4:2:0 with 16-235 range - RECOMMENDED for most cameras
    /// - **420f** (Full Range): YUV 4:2:0 with 0-255 range - Good alternative, more bandwidth
    /// - **32BGRA**: 32-bit RGBA - Highest quality but 4x bandwidth of YUV
    /// - **2vuy/yuvs**: YUV 4:2:2 formats - Higher chroma resolution than 4:2:0
    /// 
    /// - Parameter formatDescription: The format description
    /// - Returns: Human-readable pixel format string
    private func pixelFormatString(from formatDescription: CMFormatDescription) -> String {
        let pixelFormat = CMFormatDescriptionGetMediaSubType(formatDescription)
        
        switch pixelFormat {
        case kCVPixelFormatType_32BGRA: return "32BGRA"
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return "420v"  // PREFERRED
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return "420f"   // GOOD
        case kCVPixelFormatType_422YpCbCr8: return "2vuy"
        case kCVPixelFormatType_422YpCbCr8_yuvs: return "yuvs"
        default:
            // Convert FourCC to string
            let bytes = withUnsafeBytes(of: pixelFormat.bigEndian) { Array($0) }
            return String(bytes: bytes, encoding: .ascii) ?? "Unknown(\(pixelFormat))"
        }
    }
    
    /// Gets detailed status information for the current UVC device configuration
    /// - Returns: Detailed status string for UI display
    func getDetailedStatus() -> String {
        guard let device = captureDevice,
              let config = activeConfiguration else {
            return "No UVC camera connected"
        }
        
        let deviceName = device.localizedName
        let modelInfo = device.modelID ?? "Unknown Model"
        
        // Current format information
        let dimensions = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
        let currentResolution = "\(dimensions.width)×\(dimensions.height)"
        let pixelFormat = pixelFormatString(from: device.activeFormat.formatDescription)
        
        // Frame rate information
        let frameRateRange = device.activeFormat.videoSupportedFrameRateRanges.first
        let currentFrameRate = String(format: "%.0f", config.targetFrameRate)
        let maxSupportedFPS = String(format: "%.0f", frameRateRange?.maxFrameRate ?? 30.0)
        
        // Zoom information
        let currentZoom = String(format: "%.1f", device.videoZoomFactor)
        let zoomRange = "\(String(format: "%.1f", device.minAvailableVideoZoomFactor))-\(String(format: "%.1f", device.maxAvailableVideoZoomFactor))"
        
        // Format preference explanation
        let formatNote: String
        if pixelFormat == "420v" {
            formatNote = "✅ Optimal format (Video Range YUV)"
        } else if pixelFormat == "420f" {
            formatNote = "⚡ Good format (Full Range YUV)"
        } else if pixelFormat == "32BGRA" {
            formatNote = "🎨 High quality (4x bandwidth)"
        } else {
            formatNote = "⚠️ Non-standard format"
        }
        
        return """
        Camera: \(deviceName)
        Model: \(modelInfo)
        
        📺 Resolution: \(currentResolution) (\(pixelFormat))
        🎬 Frame Rate: \(currentFrameRate) fps (max: \(maxSupportedFPS))
        🔍 Zoom: \(currentZoom)x (range: \(zoomRange)x)
        
        \(formatNote)
        """
    }
    
    /// Logs comprehensive device capabilities for debugging
    /// - Parameter capabilities: The analyzed capabilities
    private func logDeviceCapabilities(_ capabilities: UVCCapabilities) {
        print("UVC: 📊 DEVICE CAPABILITIES REPORT")
        print("UVC: ═══════════════════════════════")
        print("UVC: Device: \(capabilities.deviceName)")
        print("UVC: Model: \(capabilities.modelID ?? "Unknown")")
        print("UVC: ")
        
        print("UVC: 📐 AVAILABLE RESOLUTIONS & FRAME RATES:")
        for (index, format) in capabilities.availableFormats.enumerated() {
            let res = format.resolution
            let frameRateList = format.frameRates.map { String(format: "%.0f", $0) }.joined(separator: ", ")
            print("UVC:   \(index + 1). \(Int(res.width))×\(Int(res.height)) @ [\(frameRateList)] fps (\(format.pixelFormat))")
        }
        
        print("UVC: ")
        print("UVC: 🔍 ZOOM CAPABILITIES:")
        print("UVC:   Range: \(String(format: "%.1f", capabilities.zoomRange.min))x - \(String(format: "%.1f", capabilities.zoomRange.max))x")
        
        print("UVC: ")
        print("UVC: ⚙️ SUPPORTED SESSION PRESETS:")
        let presetNames = capabilities.supportedPresets.map { presetName($0) }.joined(separator: ", ")
        print("UVC:   [\(presetNames)]")
        
        print("UVC: ")
        print("UVC: ✅ CURRENT ACTIVE FORMAT:")
        let active = capabilities.activeFormat
        print("UVC:   \(Int(active.resolution.width))×\(Int(active.resolution.height)) @ \(String(format: "%.1f", active.frameRate)) fps (\(active.pixelFormat))")
        
        print("UVC: ═══════════════════════════════")
    }
    
    /// Helper to get human-readable preset names
    /// - Parameter preset: The session preset
    /// - Returns: Human-readable name
    private func presetName(_ preset: AVCaptureSession.Preset) -> String {
        switch preset {
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .hd1280x720: return "HD720p"
        case .hd1920x1080: return "HD1080p"
        case .hd4K3840x2160: return "4K"
        case .photo: return "Photo"
        default: return "Custom"
        }
    }

    // MARK: - Format Selection & Configuration
    
    /// Finds the optimal format using priority-based fallback logic
    private func findOptimalFormat(device: AVCaptureDevice, config: UVCConfiguration) -> AVCaptureDevice.Format? {
        // Priority 1: Exact resolution + target FPS
        if let exactMatch = findFormatWithExactResolutionAndFPS(device: device, config: config, targetFPS: config.targetFrameRate) {
            let dimensions = CMVideoFormatDescriptionGetDimensions(exactMatch.formatDescription)
            let pixelFormatStr = pixelFormatString(from: exactMatch.formatDescription)
            print("UVC: Found exact match: \(dimensions.width)×\(dimensions.height) @ \(config.targetFrameRate) fps (\(pixelFormatStr))")
            return exactMatch
        }
        
        // Priority 2: Target resolution + any available FPS
        if let resolutionMatch = findFormatWithExactResolutionAnyFPS(device: device, config: config) {
            let dimensions = CMVideoFormatDescriptionGetDimensions(resolutionMatch.formatDescription)
            let pixelFormatStr = pixelFormatString(from: resolutionMatch.formatDescription)
            let maxFPS = resolutionMatch.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 30.0
            print("UVC: Found resolution match: \(dimensions.width)×\(dimensions.height) @ \(maxFPS) fps (\(pixelFormatStr))")
            return resolutionMatch
        }
        
        // Priority 3: Device default
        let currentFormat = device.activeFormat
        let dimensions = CMVideoFormatDescriptionGetDimensions(currentFormat.formatDescription)
        let pixelFormatStr = pixelFormatString(from: currentFormat.formatDescription)
        print("UVC: Using device default: \(dimensions.width)×\(dimensions.height) (\(pixelFormatStr))")
        return currentFormat
    }
    
    /// Finds format with exact resolution and frame rate match
    private func findFormatWithExactResolutionAndFPS(device: AVCaptureDevice, config: UVCConfiguration, targetFPS: Double) -> AVCaptureDevice.Format? {
        var bestFormat: AVCaptureDevice.Format?
        var bestPixelFormatScore = -1
        
        for format in device.formats {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let resolution = CGSize(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height))
            
            if resolution.width == config.targetResolution.width && resolution.height == config.targetResolution.height {
                let supportsTargetFPS = format.videoSupportedFrameRateRanges.contains { range in
                    targetFPS >= Double(range.minFrameRate) && targetFPS <= Double(range.maxFrameRate)
                }
                
                if supportsTargetFPS {
                    let pixelFormat = CMFormatDescriptionGetMediaSubType(format.formatDescription)
                    let pixelFormatScore = getPixelFormatScore(pixelFormat, preferred: config.preferredPixelFormat)
                    
                    if pixelFormatScore > bestPixelFormatScore {
                        bestPixelFormatScore = pixelFormatScore
                        bestFormat = format
                    }
                }
            }
        }
        
        return bestFormat
    }
    
    /// Finds format with exact resolution match (any frame rate)
    private func findFormatWithExactResolutionAnyFPS(device: AVCaptureDevice, config: UVCConfiguration) -> AVCaptureDevice.Format? {
        var bestFormat: AVCaptureDevice.Format?
        var bestScore = -1.0
        
        for format in device.formats {
            let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
            let resolution = CGSize(width: CGFloat(dimensions.width), height: CGFloat(dimensions.height))
            
            if resolution.width == config.targetResolution.width && resolution.height == config.targetResolution.height {
                let maxFPS = format.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 30.0
                let pixelFormat = CMFormatDescriptionGetMediaSubType(format.formatDescription)
                let pixelFormatScore = getPixelFormatScore(pixelFormat, preferred: config.preferredPixelFormat)
                let totalScore = Double(pixelFormatScore) + Double(maxFPS) * 0.1
                
                if totalScore > bestScore {
                    bestScore = totalScore
                    bestFormat = format
                }
            }
        }
        
        return bestFormat
    }
    
    /// Scores pixel formats based on preference (higher score = better)
    private func getPixelFormatScore(_ pixelFormat: OSType, preferred: OSType?) -> Int {
        // If we have a preferred format and this matches, highest score
        if let preferred = preferred, pixelFormat == preferred {
            return 100
        }
        
        // Score common formats
        switch pixelFormat {
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:  // 420f
            return 80
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: // 420v
            return 70
        case kCVPixelFormatType_422YpCbCr8:                   // 2vuy
            return 60
        case kCVPixelFormatType_422YpCbCr8_yuvs:              // yuvs
            return 50
        case kCVPixelFormatType_32BGRA:                       // 32BGRA
            return 40
        default:
            return 10  // Unknown format
        }
    }
    
    /// Applies UVC configuration with priority-based format selection
    private func applyUVCConfigurationWithFallbacks(device: AVCaptureDevice, config: UVCConfiguration) -> (success: Bool, appliedConfig: UVCConfiguration) {
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            
            // Apply optimal format
            if let selectedFormat = findOptimalFormat(device: device, config: config) {
                device.activeFormat = selectedFormat
            }
            
            // Configure frame rate
            let appliedFormat = device.activeFormat
            let appliedDimensions = CMVideoFormatDescriptionGetDimensions(appliedFormat.formatDescription)
            let appliedResolution = CGSize(width: CGFloat(appliedDimensions.width), height: CGFloat(appliedDimensions.height))
            
            var appliedFrameRate = config.targetFrameRate
            let frameDuration = CMTime(value: 1, timescale: Int32(config.targetFrameRate))
            
            if appliedFormat.videoSupportedFrameRateRanges.contains(where: { range in
                config.targetFrameRate >= Double(range.minFrameRate) && config.targetFrameRate <= Double(range.maxFrameRate)
            }) {
                device.activeVideoMinFrameDuration = frameDuration
                device.activeVideoMaxFrameDuration = frameDuration
            } else {
                let maxSupportedFPS = appliedFormat.videoSupportedFrameRateRanges.first?.maxFrameRate ?? 30.0
                appliedFrameRate = Double(maxSupportedFPS)
                let fallbackDuration = CMTime(value: 1, timescale: Int32(maxSupportedFPS))
                device.activeVideoMinFrameDuration = fallbackDuration
                device.activeVideoMaxFrameDuration = fallbackDuration
            }
            
            // Configure zoom
            var appliedZoom = config.targetZoomFactor
            if config.useWidestFOV {
                appliedZoom = device.minAvailableVideoZoomFactor
                device.videoZoomFactor = appliedZoom
            }
            
            // Configure focus and exposure
            if device.isFocusModeSupported(.continuousAutoFocus) && device.isFocusPointOfInterestSupported {
                device.focusMode = .continuousAutoFocus
                device.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            }
            
            if device.isExposureModeSupported(.continuousAutoExposure) {
                device.exposureMode = .continuousAutoExposure
            }
            
            let finalConfig = UVCConfiguration(
                targetResolution: appliedResolution,
                targetFrameRate: appliedFrameRate,
                targetZoomFactor: appliedZoom,
                useWidestFOV: config.useWidestFOV,
                preferredPixelFormat: config.preferredPixelFormat
            )
            
            return (success: true, appliedConfig: finalConfig)
            
        } catch {
            print("UVC: Configuration failed: \(error)")
            return (success: false, appliedConfig: config)
        }
    }
    
    // MARK: - FrameSource Protocol Implementation
    
    @MainActor
    func setUp(completion: @escaping (Bool) -> Void) {
        print("UVC: Starting setup...")
        
        guard let uvcDevice = Self.bestUVCDevice() else {
            print("UVC: No external cameras found during setup")
            completion(false)
            return
        }
        
        self.captureDevice = uvcDevice
        print("UVC: Selected device for setup: \(uvcDevice.localizedName)")
        
        // Apply device-specific workarounds
        applyDeviceSpecificWorkarounds(device: uvcDevice)
        
        // Analyze device capabilities first
        let capabilities = analyzeUVCCapabilities(device: uvcDevice)
        cachedCapabilities = capabilities
        
        // Apply preferred configuration with individual fallbacks
        let preferredConfig = UVCConfiguration.preferred
        
        Task {
            let success = await self.setUpUVCCameraWithConfiguration(
                device: uvcDevice,
                preferredConfig: preferredConfig
            )
            print("UVC: Setup completed with success: \(success)")
            completion(success)
        }
    }
    
    func setUpUVCCameraWithConfiguration(device: AVCaptureDevice, preferredConfig: UVCConfiguration) async -> Bool {
        print("UVC: Beginning camera configuration...")
        captureSession.beginConfiguration()
        
        // Apply configuration with individual setting fallbacks
        print("UVC: 🎯 Attempting configuration: 720p @ 60fps, widest FOV")
        let configResult = applyUVCConfigurationWithFallbacks(device: device, config: preferredConfig)
        
        if !configResult.success {
            print("UVC: ❌ Configuration failed completely")
            captureSession.commitConfiguration()
            return false
        }
        
        // Store the actually applied configuration
        self.activeConfiguration = configResult.appliedConfig
        let appliedConfig = configResult.appliedConfig
        
        await MainActor.run {
            let deviceType = UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone"
            print("UVC: ✅ \(deviceType) configured: \(Int(appliedConfig.targetResolution.width))×\(Int(appliedConfig.targetResolution.height)) @ \(appliedConfig.targetFrameRate) fps, zoom: \(String(format: "%.1f", appliedConfig.targetZoomFactor))x")
        }

        // Create device input
        do {
            print("UVC: Creating device input for \(device.localizedName)")
            videoInput = try AVCaptureDeviceInput(device: device)
            print("UVC: Device input created successfully")
        } catch {
            print("UVC: Failed to create device input: \(error)")
            captureSession.commitConfiguration()
            return false
        }

        guard captureSession.canAddInput(videoInput!) else {
            print("UVC: Cannot add video input to session")
            captureSession.commitConfiguration()
            return false
        }
        captureSession.addInput(videoInput!)
        
        // Configure preview layer
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .landscapeRight
        self.previewLayer = previewLayer

        // Configure video output to use native device format
        videoOutput.videoSettings = nil
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
        
        guard captureSession.canAddOutput(videoOutput) else {
            print("UVC: Cannot add video output to session")
            captureSession.commitConfiguration()
            return false
        }
        captureSession.addOutput(videoOutput)

        // Configure video connections without mirroring
        let connection = videoOutput.connection(with: AVMediaType.video)
        connection?.videoOrientation = .landscapeRight
        
        if let conn = connection, conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = false
        }
        
        if let previewConn = previewLayer.connection, previewConn.isVideoMirroringSupported {
            previewConn.automaticallyAdjustsVideoMirroring = false
            previewConn.isVideoMirrored = false
        }

        captureSession.commitConfiguration()
        
        // Re-apply device format after session commit (AVFoundation may reset it)
        let reapplyResult = applyUVCConfigurationWithFallbacks(device: device, config: preferredConfig)
        if reapplyResult.success {
            self.activeConfiguration = reapplyResult.appliedConfig
        }
        
        // Verify final format
        let finalFormat = device.activeFormat
        let finalDimensions = CMVideoFormatDescriptionGetDimensions(finalFormat.formatDescription)
        let finalPixelFormat = pixelFormatString(from: finalFormat.formatDescription)
        print("UVC: Active format: \(finalDimensions.width)×\(finalDimensions.height) (\(finalPixelFormat))")
        
        return true
    }

    /// Starts frame acquisition from the UVC camera
    nonisolated func start() {
        print("UVC: Start method called")
        guard !captureSession.isRunning else {
            print("UVC: Session already running")
            return
        }
        
        print("UVC: Session not running, starting...")
        DispatchQueue.global().async {
            self.captureSession.startRunning()
            print("UVC: Session startRunning() called")
            
            DispatchQueue.main.async {
                print("UVC: Session running status: \(self.captureSession.isRunning)")
            }
        }
    }

    /// Stops frame acquisition from the UVC camera
    nonisolated func stop() {
        guard captureSession.isRunning else { return }
        DispatchQueue.global().async {
            self.captureSession.stopRunning()
        }
    }

    /// Sets the zoom level for the UVC camera, if supported
    /// - Parameter ratio: The zoom ratio to apply
    nonisolated func setZoomRatio(ratio: CGFloat) {
        guard let captureDevice = captureDevice else { return }
        
        do {
            try captureDevice.lockForConfiguration()
            defer { captureDevice.unlockForConfiguration() }
            
            if captureDevice.videoZoomFactor != ratio && 
               ratio >= captureDevice.minAvailableVideoZoomFactor &&
               ratio <= captureDevice.maxAvailableVideoZoomFactor {
                captureDevice.videoZoomFactor = ratio
            }
        } catch {
            print("UVC: Zoom configuration failed: \(error)")
        }
    }
    
    /// Resets processing state to allow normal inference to resume after calibration
    @MainActor
    func resetProcessingState() {
        isModelProcessing = false
        print("UVC: Processing state reset - ready for normal inference")
    }
    
    /// Captures a still image from the UVC camera (not supported)
    /// - Parameter completion: Callback with nil (UVC cameras don't support photo capture)
    @MainActor
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void) {
        completion(nil)
    }
    
    /// Shows UI for selecting UVC content with capability reporting
    /// - Parameters:
    ///   - viewController: The view controller to present UI from
    ///   - completion: Called when selection is complete with true
    @MainActor
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        // Show capability report popup if device is available
        if let capabilities = cachedCapabilities {
            showCapabilityReport(capabilities, from: viewController)
        } else if let device = captureDevice {
            let capabilities = analyzeUVCCapabilities(device: device)
            showCapabilityReport(capabilities, from: viewController)
        }
        completion(true)
    }
    
    /// Shows a capability report popup for the connected UVC device
    /// - Parameters:
    ///   - capabilities: The device capabilities to display
    ///   - viewController: The view controller to present from
    @MainActor
    private func showCapabilityReport(_ capabilities: UVCCapabilities, from viewController: UIViewController) {
        let alert = UIAlertController(
            title: "📹 UVC Camera Detected",
            message: generateCapabilityMessage(capabilities),
            preferredStyle: .alert
        )
        
        alert.addAction(UIAlertAction(title: "Continue", style: .default))
        viewController.present(alert, animated: true)
    }
    
    /// Generates a user-friendly capability message
    /// - Parameter capabilities: The device capabilities
    /// - Returns: Formatted capability string
    private func generateCapabilityMessage(_ capabilities: UVCCapabilities) -> String {
        var message = "Device: \(capabilities.deviceName)\n"
        
        if let modelID = capabilities.modelID {
            message += "Model: \(modelID)\n"
        }
        
        message += "\n📐 Resolution Support:\n"
        let topFormats = Array(capabilities.availableFormats.prefix(3))
        for format in topFormats {
            let res = format.resolution
            let maxFPS = format.frameRates.max() ?? 0
            message += "• \(Int(res.width))×\(Int(res.height)) @ \(Int(maxFPS)) fps\n"
        }
        
        if capabilities.availableFormats.count > 3 {
            message += "• + \(capabilities.availableFormats.count - 3) more formats\n"
        }
        
        message += "\n🔍 Zoom Range: \(String(format: "%.1f", capabilities.zoomRange.min))x - \(String(format: "%.1f", capabilities.zoomRange.max))x\n"
        
        message += "\n✅ Current Config:\n"
        let active = capabilities.activeFormat
        message += "\(Int(active.resolution.width))×\(Int(active.resolution.height)) @ \(String(format: "%.0f", active.frameRate)) fps"
        
        return message
    }
    
    /// Requests camera permission for UVC cameras
    /// - Parameter completion: Called with the result of the permission request
    @MainActor
    func requestPermission(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }
    
    /// Updates the UVC camera for orientation changes with smooth transitions
    /// - Parameter orientation: The new device orientation
    @MainActor
    func updateForOrientationChange(orientation: UIDeviceOrientation) {
        guard !isUpdatingOrientation else {
            pendingOrientationUpdate = orientation
            return
        }
        
        guard orientation != lastKnownOrientation else { return }
        
        let shouldUpdateVideoSystem: Bool
        switch orientation {
        case .portrait, .portraitUpsideDown, .landscapeLeft, .landscapeRight:
            lastKnownOrientation = orientation
            shouldUpdateVideoSystem = true
        case .faceUp, .faceDown, .unknown:
            if frameProcessingCount % 600 == 0 {
                print("UVC: Device flat orientation (\(orientation.rawValue)), maintaining last known: \(lastKnownOrientation.rawValue)")
            }
            shouldUpdateVideoSystem = false
        default:
            shouldUpdateVideoSystem = false
        }
        
        guard shouldUpdateVideoSystem else { return }
        
        print("UVC: Orientation change - device: \(orientation.rawValue) → video: landscapeRight")
        isUpdatingOrientation = true
        
        Task {
            await MainActor.run {
                self.setVideoGravityForOrientation(orientation: orientation)
            }
            
            await Task.detached {
                await MainActor.run {
                    self.updateVideoOrientationSmoothly(orientation: .landscapeRight, deviceOrientation: orientation)
                }
            }.value
            
            await MainActor.run {
                self.isUpdatingOrientation = false
                
                if let pendingOrientation = self.pendingOrientationUpdate {
                    self.pendingOrientationUpdate = nil
                    if pendingOrientation != orientation {
                        self.updateForOrientationChange(orientation: pendingOrientation)
                    }
                }
            }
        }
    }
    
    /// Sets video gravity for the given orientation with smooth animation
    /// - Parameter orientation: The device orientation
    @MainActor
    private func setVideoGravityForOrientation(orientation: UIDeviceOrientation) {
        guard let previewLayer = self.previewLayer else { return }
        
        let newVideoGravity: AVLayerVideoGravity = orientation.isPortrait ? .resizeAspect : .resizeAspectFill
        
        guard previewLayer.videoGravity != newVideoGravity else { return }
        
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.3)
        CATransaction.setAnimationTimingFunction(CAMediaTimingFunction(name: .easeInEaseOut))
        
        previewLayer.videoGravity = newVideoGravity
        print("UVC: Video gravity smoothly updated to \(newVideoGravity.rawValue)")
        
        CATransaction.commit()
    }
    
    /// Updates video orientation smoothly with animation
    /// - Parameters:
    ///   - orientation: The video orientation to set
    ///   - deviceOrientation: The current device orientation
    @MainActor
    func updateVideoOrientationSmoothly(orientation: AVCaptureVideoOrientation, deviceOrientation: UIDeviceOrientation) {
        guard let connection = videoOutput.connection(with: .video) else {
            print("UVC: No video connection available for orientation update")
            return
        }
        
        guard connection.videoOrientation != orientation else { return }
        
        CATransaction.begin()
        CATransaction.setAnimationDuration(0.25)
        CATransaction.setAnimationTimingFunction(CAMediaTimingFunction(name: .easeInEaseOut))
        
        connection.videoOrientation = orientation
        
        // No mirroring for UVC cameras to maintain consistency
        if connection.isVideoMirroringSupported && connection.isVideoMirrored {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
            print("UVC: Video output mirroring updated: false")
        }
        
        if let previewConnection = self.previewLayer?.connection {
            if previewConnection.videoOrientation != orientation {
                previewConnection.videoOrientation = orientation
            }
            
            if previewConnection.isVideoMirroringSupported && previewConnection.isVideoMirrored {
                previewConnection.automaticallyAdjustsVideoMirroring = false
                previewConnection.isVideoMirrored = false
                print("UVC: Preview layer mirroring updated: false")
            }
        }
        
        CATransaction.commit()
        print("UVC: Video orientation smoothly updated to \(orientation.rawValue)")
    }
    
    // MARK: - UI Integration & Coordinate Transformation
    
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        guard let previewLayer = self.previewLayer else { return }
        
        view.layer.insertSublayer(previewLayer, at: 0)
        previewLayer.frame = view.bounds
        
        let orientation = UIDevice.current.orientation
        setVideoGravityForOrientation(orientation: orientation)
        updateForOrientationChange(orientation: orientation)
        
        print("UVC: Integrated with YOLOView - frame: \(previewLayer.frame), gravity: \(previewLayer.videoGravity.rawValue)")
    }
    
    /// Adds an overlay layer to the UVC preview
    /// - Parameter layer: The layer to add
    @MainActor
    func addOverlayLayer(_ layer: CALayer) {
        previewLayer?.addSublayer(layer)
    }
    
    /// Adds bounding box views to the UVC preview
    /// - Parameter boxViews: The bounding box views to add
    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
        guard let previewLayer = self.previewLayer else { return }
        for box in boxViews {
            box.addToLayer(previewLayer)
        }
    }
    
    /// Transforms normalized detection coordinates to screen coordinates for UVC cameras
    /// - Parameters:
    ///   - rect: The normalized detection rectangle (0.0-1.0)
    ///   - viewBounds: The bounds of the view where the detection will be displayed
    ///   - orientation: The current device orientation
    /// - Returns: A rectangle in screen coordinates
    @MainActor
    func transformDetectionToScreenCoordinates(rect: CGRect, viewBounds: CGRect, orientation: UIDeviceOrientation) -> CGRect {
        let width = viewBounds.width
        let height = viewBounds.height
        
        let effectiveOrientation: UIDeviceOrientation
        switch orientation {
        case .faceUp, .faceDown, .unknown:
            effectiveOrientation = lastKnownOrientation
            if frameProcessingCount % 900 == 0 {
                print("UVC: Coordinate transform using last known orientation \(lastKnownOrientation.rawValue) (device: \(orientation.rawValue))")
            }
        default:
            effectiveOrientation = orientation
        }
        
        if effectiveOrientation.isPortrait {
            // Portrait mode with letterboxing (resizeAspect)
            let frameAspectRatio: CGFloat = 16.0 / 9.0
            let streamHeight = width / frameAspectRatio
            let blackBarHeight = (height - streamHeight) / 2
            
            let scaledY = rect.origin.y * streamHeight + blackBarHeight
            let scaledHeight = rect.size.height * streamHeight
            
            return CGRect(
                x: rect.origin.x * width,
                y: height - scaledY - scaledHeight,
                width: rect.size.width * width,
                height: scaledHeight
            )
        } else {
            // Landscape mode
            let frameAspectRatio = longSide / shortSide
            let viewAspectRatio = width / height
            var scaleX: CGFloat = 1.0
            var scaleY: CGFloat = 1.0
            var offsetX: CGFloat = 0.0
            var offsetY: CGFloat = 0.0
            
            if frameAspectRatio > viewAspectRatio {
                scaleY = height / shortSide
                scaleX = scaleY
                offsetX = (longSide * scaleX - width) / 2
            } else {
                scaleX = width / longSide
                scaleY = scaleX
                offsetY = (shortSide * scaleY - height) / 2
            }
            
            return CGRect(
                x: rect.origin.x * longSide * scaleX - offsetX,
                y: height - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY),
                width: rect.size.width * longSide * scaleX,
                height: rect.size.height * shortSide * scaleY
            )
        }
    }
    
    // MARK: - Frame Processing
    
    /// Processes frames on the camera queue for inference or calibration
    /// - Parameter sampleBuffer: The sample buffer containing the frame data
    private func processFrameOnCameraQueue(sampleBuffer: CMSampleBuffer) {
        guard let predictor = predictor else {
            print("UVC: predictor is nil")
            return
        }
        
        let pipelineStartTime = CACurrentMediaTime()
        pipelineStartTimes.append(pipelineStartTime)
        if pipelineStartTimes.count > 30 {
            pipelineStartTimes.removeFirst()
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { 
            return 
        }
        
        // Capture frame dimensions on first frame
        if !frameSizeCaptured {
            let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            longSide = max(frameWidth, frameHeight)
            shortSide = min(frameWidth, frameHeight)
            frameSizeCaptured = true
            
            Task { @MainActor in
                let configDescription = self.activeConfiguration.map { 
                    "\(Int($0.targetResolution.width))×\(Int($0.targetResolution.height)) @ \(Int($0.targetFrameRate))fps" 
                } ?? "Default"
                print("UVC: Frame size: \(Int(frameWidth))×\(Int(frameHeight)) (\(configDescription))")
            }
        }
        
        frameProcessingCount += 1
        frameProcessingTimestamps.append(CACurrentMediaTime())
        if frameProcessingTimestamps.count > 60 {
            frameProcessingTimestamps.removeFirst()
        }
        
        let framePreparationStartTime = CACurrentMediaTime()
        
        // Pass frame to delegate
        if let frameSourceDelegate = frameSourceDelegate {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                let uiImage = UIImage(cgImage: cgImage)
                
                DispatchQueue.main.async {
                    self.frameSourceDelegate?.frameSource(self, didOutputImage: uiImage)
                }
            }
        }
        
        lastFramePreparationTime = (CACurrentMediaTime() - framePreparationStartTime) * 1000
        
        // Handle calibration mode
        let shouldProcessForCalibration = !inferenceOK && predictor is TrackingDetector
        
        if shouldProcessForCalibration {
            let conversionStartTime = CACurrentMediaTime()
            
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                let image = UIImage(cgImage: cgImage)
                lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
                
                DispatchQueue.main.async { [weak self] in
                    guard let self = self,
                          let trackingDetector = self.predictor as? TrackingDetector else { return }
                    
                    if trackingDetector.getCalibrationFrameCount() == 0 {
                        self.videoCaptureDelegate?.onClearBoxes()
                    }
                          
                    if let pixelBufferForCalibration = self.createStandardPixelBuffer(from: image, forSourceType: self.sourceType) {
                        trackingDetector.processFrame(pixelBufferForCalibration)
                    }
                }
            } else {
                lastConversionTime = (CACurrentMediaTime() - conversionStartTime) * 1000
            }
            
            if frameProcessingCount % 300 == 0 {
                logPipelineAnalysis(mode: "calibration")
            }
            
            return
        }
        
        // Normal inference
        if inferenceOK && !isModelProcessing {
            isModelProcessing = true
            predictor.predict(sampleBuffer: sampleBuffer, onResultsListener: self, onInferenceTime: self)
        }
        
        if frameProcessingCount % 300 == 0 {
            logPipelineAnalysis(mode: "inference")
        }
    }
    
    // MARK: - Performance Analysis
    
    /// Logs performance analysis with unified format
    /// - Parameter mode: The processing mode ("inference" or "calibration")
    private func logPipelineAnalysis(mode: String) {
        guard shouldLogPerformance(frameCount: frameProcessingCount) else { return }
        
        let processingFPS: Double
        if frameProcessingTimestamps.count >= 2 {
            let windowSeconds = frameProcessingTimestamps.last! - frameProcessingTimestamps.first!
            processingFPS = Double(frameProcessingTimestamps.count - 1) / windowSeconds
        } else {
            processingFPS = 0
        }
        
        let actualThroughputFPS = calculateThroughputFPS()
        let frameSize = frameSizeCaptured ? CGSize(width: longSide, height: shortSide) : .zero
        
        let timingData: [String: Double] = [
            "preparation": lastFramePreparationTime,
            "conversion": lastConversionTime,
            "inference": lastInferenceTime,
            "ui": lastUITime,
            "total": lastTotalPipelineTime,
            "throughput": actualThroughputFPS
        ]
        
        logUnifiedPerformanceAnalysis(
            frameCount: frameProcessingCount,
            sourcePrefix: "UVC",
            frameSize: frameSize,
            processingFPS: processingFPS,
            mode: mode,
            timingData: timingData
        )
    }
    
    /// Calculates actual throughput FPS based on pipeline completion times
    /// - Returns: The calculated throughput FPS
    private func calculateThroughputFPS() -> Double {
        guard pipelineCompleteTimes.count >= 2 else { return 0 }
        let cyclesToConsider = min(10, pipelineCompleteTimes.count)
        let recentCompletions = Array(pipelineCompleteTimes.suffix(cyclesToConsider))
        let timeInterval = recentCompletions.last! - recentCompletions.first!
        return timeInterval > 0 ? Double(cyclesToConsider - 1) / timeInterval : 0
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension UVCVideoSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        processFrameOnCameraQueue(sampleBuffer: sampleBuffer)
    }
}

// MARK: - ResultsListener & InferenceTimeListener

extension UVCVideoSource: ResultsListener, InferenceTimeListener {
    func on(inferenceTime: Double, fpsRate: Double) {
        lastInferenceTime = inferenceTime
        
        DispatchQueue.main.async {
            self.videoCaptureDelegate?.onInferenceTime(speed: inferenceTime, fps: fpsRate)
            self.frameSourceDelegate?.frameSource(self, didUpdateWithSpeed: inferenceTime, fps: fpsRate)
        }
    }

    func on(result: YOLOResult) {
        let postProcessingStartTime = CACurrentMediaTime()
        let timestamp = CACurrentMediaTime()
        
        isModelProcessing = false
        
        frameTimestamps.append(timestamp)
        if frameTimestamps.count > 30 {
            frameTimestamps.removeFirst()
        }
        
        let uiDelegateStartTime = CACurrentMediaTime()
        
        DispatchQueue.main.async {
            self.videoCaptureDelegate?.onPredict(result: result)
        }
        
        lastUITime = (CACurrentMediaTime() - uiDelegateStartTime) * 1000
        let totalUITime = (CACurrentMediaTime() - postProcessingStartTime) * 1000
        lastTotalPipelineTime = lastFramePreparationTime + lastInferenceTime + totalUITime
        
        pipelineCompleteTimes.append(timestamp)
        if pipelineCompleteTimes.count > 30 {
            pipelineCompleteTimes.removeFirst()
        }
        
        let actualThroughputFPS = calculateThroughputFPS()
        DispatchQueue.main.async {
            self.frameSourceDelegate?.frameSource(self, didUpdateWithSpeed: result.speed, fps: actualThroughputFPS)
        }
    }
}

// MARK: - UIDeviceOrientation Extension

private extension UIDeviceOrientation {
    /// Returns true if the orientation is portrait or portrait upside down
    var isPortrait: Bool {
        return self == .portrait || self == .portraitUpsideDown
    }
}
