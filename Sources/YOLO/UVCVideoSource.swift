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
/// This class provides UVC (USB Video Class) camera support for iPad devices with USB-C ports.
/// It adapts the existing CameraVideoSource implementation to work with external cameras
/// discovered through AVFoundation's external device APIs introduced in iPadOS 17.
@preconcurrency
class UVCVideoSource: NSObject, FrameSource, @unchecked Sendable {
    var predictor: FrameProcessor!
    var previewLayer: AVCaptureVideoPreviewLayer?
    weak var videoCaptureDelegate: VideoCaptureDelegate?
    weak var frameSourceDelegate: FrameSourceDelegate?
    var captureDevice: AVCaptureDevice?
    let captureSession = AVCaptureSession()
    var videoInput: AVCaptureDeviceInput? = nil
    let videoOutput = AVCaptureVideoDataOutput()
    let cameraQueue = DispatchQueue(label: "uvc-camera-queue")
    var inferenceOK = true
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    var frameSizeCaptured = false
    
    // Track the last known valid orientation for coordinate transformation
    private var lastKnownOrientation: UIDeviceOrientation = .portrait
    
    // Add orientation change optimization
    private var isUpdatingOrientation = false
    private var pendingOrientationUpdate: UIDeviceOrientation?
    
    // Implement FrameSource protocol property
    var delegate: FrameSourceDelegate? {
        get { return frameSourceDelegate }
        set { frameSourceDelegate = newValue }
    }
    
    // Implement FrameSource protocol property
    var sourceType: FrameSourceType { return .uvc }

    private var currentBuffer: CVPixelBuffer?
    
    // MARK: - Performance Metrics (similar to CameraVideoSource)
    
    // Frame Processing State
    private var frameProcessingCount = 0
    private var frameProcessingTimestamps: [CFTimeInterval] = []
    private var isModelProcessing: Bool = false
    private var lastTriggerSource: String = "uvc"
    
    // Performance Timing Metrics
    private var lastFramePreparationTime: Double = 0
    private var lastConversionTime: Double = 0
    private var lastInferenceTime: Double = 0
    private var lastUITime: Double = 0
    private var lastTotalPipelineTime: Double = 0
    
    // FPS Calculation Arrays
    private var frameTimestamps: [CFTimeInterval] = []
    private var pipelineStartTimes: [CFTimeInterval] = []
    private var pipelineCompleteTimes: [CFTimeInterval] = []
    
    // MARK: - UVC-Specific Configuration
    
    /// Determines the optimal session preset for UVC cameras
    /// UVC cameras may have different capabilities than built-in cameras
    @MainActor
    private var optimalSessionPreset: AVCaptureSession.Preset {
        // Start with HD 720p for optimal YOLO processing, but check device support
        if let device = captureDevice, device.supportsSessionPreset(.hd1280x720) {
            return .hd1280x720
        } else if let device = captureDevice, device.supportsSessionPreset(.hd1920x1080) {
            return .hd1920x1080
        } else {
            // Fallback to medium quality if HD not supported
            return .medium
        }
    }
    
    /// Gets a human-readable description of the current session preset
    @MainActor
    private var sessionPresetDescription: String {
        switch optimalSessionPreset {
        case .hd1280x720:
            return "HD 1280×720 (UVC Optimized)"
        case .hd1920x1080:
            return "Full HD 1920×1080 (UVC)"
        case .medium:
            return "Medium Quality (UVC Fallback)"
        default:
            return "UVC Default"
        }
    }
    
    // MARK: - Connection Monitoring (Apple WWDC 2023 Recommendations)
    
    /// Discovery session for monitoring device connections
    private var deviceDiscoverySession: AVCaptureDevice.DiscoverySession?
    
    /// Observer for device connection changes
    private var deviceConnectionObserver: NSKeyValueObservation?
    
    // Default initializer with connection monitoring setup
    override init() {
        super.init()
        // Setup connection monitoring asynchronously to avoid main actor isolation issues
        Task { @MainActor in
            self.setupConnectionMonitoring()
        }
    }
    
    deinit {
        cleanupConnectionMonitoring()
    }
    
    /// Sets up connection monitoring for external cameras as per Apple WWDC 2023 recommendations
    @MainActor
    private func setupConnectionMonitoring() {
        guard #available(iOS 17.0, *) else { return }
        
        // Create discovery session for monitoring
        deviceDiscoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external],
            mediaType: .video,
            position: .unspecified
        )
        
        // Monitor device list changes
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
    
    /// Cleans up connection monitoring
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
        
        // Check if our current device is still connected
        if let currentDevice = captureDevice {
            let isStillConnected = currentDevices.contains { device in
                device.uniqueID == currentDevice.uniqueID && device.isConnected
            }
            
            if !isStillConnected {
                print("UVC: Current device \(currentDevice.localizedName) has been disconnected")
                handleDeviceDisconnection()
            }
        }
    }
    
    /// Handles device disconnection
    @MainActor
    private func handleDeviceDisconnection() {
        print("UVC: Handling device disconnection")
        
        // Stop the current session
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
        
        // Clear device reference
        captureDevice = nil
        
        // Notify delegates about disconnection if needed
        if let delegate = videoCaptureDelegate as? YOLOView {
            // Could notify YOLOView to switch back to camera source
            print("UVC: Device disconnected - consider switching back to camera source")
        }
    }

    // MARK: - UVC Device Discovery
    
    /// Discovers available UVC (external) cameras with basic validation
    /// Note: For production use, prefer discoverUVCCamerasWithCleanup() for rigorous validation
    /// - Returns: Array of available external camera devices
    @MainActor
    static func discoverUVCCameras() -> [AVCaptureDevice] {
        print("UVC: Starting basic device discovery...")
        
        // Check iOS version compatibility for external cameras
        if #available(iOS 17.0, *) {
            print("UVC: iOS 17.0+ detected, searching for external cameras...")
            
            // Use discovery session to find external cameras
            let discoverySession = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.external],
                mediaType: .video,
                position: .unspecified
            )
            
            let allDevices = discoverySession.devices
            print("UVC: Found \(allDevices.count) total devices in discovery session")
            
            for (index, device) in allDevices.enumerated() {
                print("UVC: Device \(index): \(device.localizedName) - Type: \(device.deviceType.rawValue)")
                print("UVC: Device \(index): Has video: \(device.hasMediaType(.video))")
                print("UVC: Device \(index): Position: \(device.position.rawValue)")
                print("UVC: Device \(index): Connected: \(device.isConnected)")
            }
            
            // Apply basic filtering
            let filteredDevices = allDevices.filter { device in
                let hasVideo = device.hasMediaType(.video)
                let isConnected = device.isConnected
                let isExternal = device.deviceType == .external
                
                let isValid = hasVideo && isConnected && isExternal
                print("UVC: Basic filtering - \(device.localizedName): \(isValid ? "VALID" : "INVALID")")
                return isValid
            }
            
            print("UVC: Basic filtering complete - \(filteredDevices.count) suitable external cameras")
            return filteredDevices
        } else {
            // Fallback for iOS < 17.0 - return empty array
            print("UVC: External cameras require iOS 17.0 or later (current iOS version not supported)")
            return []
        }
    }
    


    
    /// Gets the best available UVC camera device
    /// - Returns: The first available external camera, or nil if none found
    @MainActor
    static func bestUVCDevice() -> AVCaptureDevice? {
        let uvcCameras = discoverUVCCameras()
        
        // Log available UVC devices
        print("UVC: Found \(uvcCameras.count) external camera(s)")
        for (index, camera) in uvcCameras.enumerated() {
            print("UVC: Device \(index): \(camera.localizedName)")
        }
        
        // Return the first available UVC camera
        let bestDevice = uvcCameras.first
        if let device = bestDevice {
            print("UVC: Selected best device: \(device.localizedName) (Model: \(device.modelID ?? "Unknown"))")
        } else {
            print("UVC: No external cameras found")
        }
        
        return bestDevice
    }



    // MARK: - Setup Methods
    
    func setUp(
        sessionPreset: AVCaptureSession.Preset? = nil,
        orientation: UIDeviceOrientation,
        completion: @escaping @MainActor @Sendable (Bool) -> Void
    ) {
        print("UVC: Starting setup...")
        
        Task { @MainActor in
            // Discover UVC devices
            guard let uvcDevice = Self.bestUVCDevice() else {
                print("UVC: No external cameras found during setup")
                completion(false)
                return
            }
            
            self.captureDevice = uvcDevice
            print("UVC: Selected device for setup: \(uvcDevice.localizedName)")
            
            let preset = sessionPreset ?? self.optimalSessionPreset
            
            // Move to background queue for camera setup
            Task.detached {
                let success = await self.setUpUVCCamera(
                    sessionPreset: preset, 
                    orientation: orientation)
                
                // Call completion on main queue
                await MainActor.run {
                    print("UVC: Setup completed with success: \(success)")
                    completion(success)
                }
            }
        }
    }
    
    // Simplified version for FrameSource protocol compatibility
    @MainActor
    func setUp(completion: @escaping @Sendable (Bool) -> Void) {
        print("UVC: Starting simplified setup...")
        let orientation = UIDevice.current.orientation

        // Discover UVC devices
        guard let uvcDevice = Self.bestUVCDevice() else {
            print("UVC: No external cameras found during simplified setup")
            completion(false)
            return
        }
        
        self.captureDevice = uvcDevice
        print("UVC: Selected device for simplified setup: \(uvcDevice.localizedName)")
        
        let preset = self.optimalSessionPreset
        
        Task.detached {
            let success = await self.setUpUVCCamera(
                sessionPreset: preset, 
                orientation: orientation)
            
            await MainActor.run {
                print("UVC: Simplified setup completed with success: \(success)")
                completion(success)
            }
        }
    }

    func setUpUVCCamera(
        sessionPreset: AVCaptureSession.Preset,
        orientation: UIDeviceOrientation
    ) async -> Bool {
        print("UVC: Beginning camera configuration...")
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset
        
        guard let captureDevice = captureDevice else {
            print("UVC: No capture device available")
            captureSession.commitConfiguration()
            return false
        }
        
        // Log device configuration
        await MainActor.run {
            let deviceType = UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone"
            let presetDescription = self.sessionPresetDescription
            print("UVC: Configuring \(deviceType) with \(presetDescription)")
        }

        do {
            print("UVC: Creating device input for \(captureDevice.localizedName)")
            videoInput = try AVCaptureDeviceInput(device: captureDevice)
            print("UVC: Device input created successfully")
        } catch {
            print("UVC: Failed to create device input: \(error)")
            captureSession.commitConfiguration()
            return false
        }

        if captureSession.canAddInput(videoInput!) {
            captureSession.addInput(videoInput!)
            print("UVC: Video input added to session successfully")
        } else {
            print("UVC: Cannot add video input to session")
            captureSession.commitConfiguration()
            return false
        }
        
        // Configure video orientation using UVC-specific mapping
        var videoOrientation = AVCaptureVideoOrientation.portrait
        switch orientation {
        case .portrait:
            // Rotate -90 degrees: landscape stream → portrait display
            videoOrientation = .landscapeRight
        case .portraitUpsideDown:
            // Rotate +90 degrees: landscape stream → upside down portrait display (180° from portrait)
            videoOrientation = .landscapeLeft
        case .landscapeLeft:
            // Rotate 180 degrees: flip the landscape stream
            videoOrientation = .landscapeRight
        case .landscapeRight:
            // Rotate 180 degrees: flip the landscape stream
            videoOrientation = .landscapeLeft
        default:
            videoOrientation = .landscapeRight  // Default to portrait mode mapping
        }
        
        print("UVC: Creating preview layer...")
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        // Set initial video gravity based on orientation
        if orientation == .portrait || orientation == .portraitUpsideDown {
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspect  // Letterboxing for portrait
        } else {
            previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill  // Fill for landscape
        }
        previewLayer.connection?.videoOrientation = videoOrientation
        self.previewLayer = previewLayer
        print("UVC: Preview layer created with gravity: \(previewLayer.videoGravity.rawValue)")

        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]

        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
        
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        } else {
            print("UVC: Cannot add video output to session")
            captureSession.commitConfiguration()
            return false
        }

        // Configure video connection
        let connection = videoOutput.connection(with: AVMediaType.video)
        connection?.videoOrientation = videoOrientation
        
        // Configure mirroring for UVC cameras based on orientation
        if let conn = connection, conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            // Mirror for portrait modes only, not for landscape
            conn.isVideoMirrored = (orientation == .portrait || orientation == .portraitUpsideDown)
            print("UVC: Video output mirroring configured during setup: \(conn.isVideoMirrored)")
        }
        
        // Configure preview layer connection mirroring
        if let previewConn = previewLayer.connection, previewConn.isVideoMirroringSupported {
            previewConn.automaticallyAdjustsVideoMirroring = false
            // Mirror for portrait modes only, not for landscape
            previewConn.isVideoMirrored = (orientation == .portrait || orientation == .portraitUpsideDown)
            print("UVC: Preview layer mirroring configured during setup: \(previewConn.isVideoMirrored)")
        }

        // Configure UVC device settings if supported
        do {
            try captureDevice.lockForConfiguration()
            
            // Configure focus if supported
            if captureDevice.isFocusModeSupported(AVCaptureDevice.FocusMode.continuousAutoFocus),
               captureDevice.isFocusPointOfInterestSupported {
                captureDevice.focusMode = AVCaptureDevice.FocusMode.continuousAutoFocus
                captureDevice.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            }
            
            // Configure exposure if supported
            if captureDevice.isExposureModeSupported(.continuousAutoExposure) {
                captureDevice.exposureMode = AVCaptureDevice.ExposureMode.continuousAutoExposure
            }
            
            captureDevice.unlockForConfiguration()
        } catch {
            print("UVC: Device configuration failed: \(error)")
        }

        print("UVC: Committing session configuration...")
        captureSession.commitConfiguration()
        print("UVC: Session configuration committed successfully")
        return true
    }

    // MARK: - Control Methods (same as CameraVideoSource)
    
    nonisolated func start() {
        print("UVC: Start method called")
        if !captureSession.isRunning {
            print("UVC: Session not running, starting...")
            DispatchQueue.global().async {
                self.captureSession.startRunning()
                print("UVC: Session startRunning() called")
                
                DispatchQueue.main.async {
                    print("UVC: Session running status: \(self.captureSession.isRunning)")
                }
            }
        } else {
            print("UVC: Session already running")
        }
    }

    nonisolated func stop() {
        if captureSession.isRunning {
            DispatchQueue.global().async {
                self.captureSession.stopRunning()
            }
        }
    }

    nonisolated func setZoomRatio(ratio: CGFloat) {
        guard let captureDevice = captureDevice else { return }
        
        do {
            try captureDevice.lockForConfiguration()
            defer {
                captureDevice.unlockForConfiguration()
            }
            
            // Check if zoom is supported on this UVC device
            if captureDevice.videoZoomFactor != ratio && 
               ratio >= captureDevice.minAvailableVideoZoomFactor &&
               ratio <= captureDevice.maxAvailableVideoZoomFactor {
                captureDevice.videoZoomFactor = ratio
            }
        } catch {
            print("UVC: Zoom configuration failed: \(error)")
        }
    }
    
    // MARK: - State Management
    
    @MainActor
    func resetProcessingState() {
        isModelProcessing = false
        print("UVC: Processing state reset - ready for normal inference")
    }
    
    @MainActor
    func capturePhoto(completion: @escaping @Sendable (UIImage?) -> Void) {
        // UVC cameras don't support photo capture in this implementation
        // Return nil to indicate photo capture is not supported
        completion(nil)
    }
    
    @MainActor
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        // UVC cameras don't need content selection UI
        // They automatically connect to the first available external camera
        completion(true)
    }
    
    // MARK: - Frame Processing (adapted from CameraVideoSource)
    
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
        
        if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            // Update frame size if needed
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
                
                Task { @MainActor in
                    let presetDescription = self.sessionPresetDescription
                    print("UVC: Frame size captured: \(Int(frameWidth))×\(Int(frameHeight)) (\(presetDescription))")
                    print("UVC: Coordinate system - longSide: \(Int(self.longSide)), shortSide: \(Int(self.shortSide))")
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
    }

    // MARK: - FrameSource Protocol Implementation
    
    @MainActor
    func requestPermission(completion: @escaping (Bool) -> Void) {
        // UVC cameras use the same camera permission as built-in cameras
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
    
    @MainActor
    func updateForOrientationChange(orientation: UIDeviceOrientation) {
        // Prevent redundant orientation updates
        guard !isUpdatingOrientation else {
            // Store pending update if we're already processing one
            pendingOrientationUpdate = orientation
            return
        }
        
        // Skip update if orientation hasn't actually changed
        guard orientation != lastKnownOrientation else {
            return
        }
        
        var videoOrientation: AVCaptureVideoOrientation = .portrait
        var shouldUpdateVideoSystem = true
        
        // UVC cameras need special orientation mapping to display correctly
        // The camera stream needs to be rotated to match the expected orientation
        switch orientation {
        case .portrait:
            // Rotate -90 degrees: landscape stream → portrait display
            videoOrientation = .landscapeRight
            lastKnownOrientation = .portrait
        case .portraitUpsideDown:
            // Rotate +90 degrees: landscape stream → upside down portrait display (180° from portrait)
            videoOrientation = .landscapeLeft
            lastKnownOrientation = .portraitUpsideDown
        case .landscapeLeft:
            // Rotate 180 degrees: flip the landscape stream
            videoOrientation = .landscapeRight
            lastKnownOrientation = .landscapeLeft
        case .landscapeRight:
            // Rotate 180 degrees: flip the landscape stream
            videoOrientation = .landscapeLeft
            lastKnownOrientation = .landscapeRight
        case .faceUp, .faceDown, .unknown:
            // When device is flat (.faceUp, .faceDown, .unknown), don't change video orientation
            // Keep using the last known orientation for coordinate transformation
            // Reduce logging noise - only log occasionally for flat orientations
            if frameProcessingCount % 600 == 0 { // Log every ~20 seconds at 30fps
                print("UVC: Device flat orientation (\(orientation.rawValue)), maintaining last known: \(lastKnownOrientation.rawValue)")
            }
            shouldUpdateVideoSystem = false
        default:
            // For any other unknown orientations, maintain current state
            shouldUpdateVideoSystem = false
        }
        
        // Only proceed with video system updates if orientation is valid
        guard shouldUpdateVideoSystem else { return }
        
        print("UVC: Orientation change - device: \(orientation.rawValue) → video: \(videoOrientation.rawValue)")
        
        // Mark as updating to prevent concurrent updates
        isUpdatingOrientation = true
        
        // Perform orientation update asynchronously to avoid blocking main thread
        Task {
            // Update video gravity for the new orientation (main thread)
            await MainActor.run {
                self.setVideoGravityForOrientation(orientation: orientation)
            }
            
            // Update video orientation on background queue to avoid blocking
            await Task.detached {
                await MainActor.run {
                    self.updateVideoOrientation(orientation: videoOrientation)
                }
            }.value
            
            // Mark update as complete and handle any pending updates
            await MainActor.run {
                self.isUpdatingOrientation = false
                
                // Process any pending orientation update
                if let pendingOrientation = self.pendingOrientationUpdate {
                    self.pendingOrientationUpdate = nil
                    // Only process if it's different from what we just applied
                    if pendingOrientation != orientation {
                        self.updateForOrientationChange(orientation: pendingOrientation)
                    }
                }
            }
        }
    }
    
    @MainActor
    private func setVideoGravityForOrientation(orientation: UIDeviceOrientation) {
        guard let previewLayer = self.previewLayer else { return }
        
        let newVideoGravity: AVLayerVideoGravity
        switch orientation {
        case .portrait, .portraitUpsideDown:
            // In portrait mode, maintain aspect ratio with letterboxing (black bars)
            // This prevents the 16:9 landscape stream from being stretched/rotated
            newVideoGravity = .resizeAspect
        case .landscapeLeft, .landscapeRight:
            // In landscape mode, fill the screen since orientations match
            newVideoGravity = .resizeAspectFill
        default:
            // Default to aspect fit for unknown orientations
            newVideoGravity = .resizeAspect
        }
        
        // Only update if gravity actually changed to avoid unnecessary operations
        if previewLayer.videoGravity != newVideoGravity {
            previewLayer.videoGravity = newVideoGravity
            print("UVC: Video gravity updated to \(newVideoGravity.rawValue)")
        }
    }
    
    @MainActor
    func updateVideoOrientation(orientation: AVCaptureVideoOrientation) {
        guard let connection = videoOutput.connection(with: .video) else { 
            print("UVC: No video connection available for orientation update")
            return 
        }
        
        // Only update if orientation actually changed
        guard connection.videoOrientation != orientation else {
            return
        }
        
        connection.videoOrientation = orientation
        
        // Configure mirroring based on device orientation
        let deviceOrientation = UIDevice.current.orientation
        let shouldMirror = (deviceOrientation == .portrait || deviceOrientation == .portraitUpsideDown)
        
        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            // Only update mirroring if it changed
            if connection.isVideoMirrored != shouldMirror {
                connection.isVideoMirrored = shouldMirror
                print("UVC: Video output mirroring updated: \(shouldMirror)")
            }
        }
        
        // Update preview layer connection
        if let previewConnection = self.previewLayer?.connection {
            // Only update if orientation changed
            if previewConnection.videoOrientation != orientation {
                previewConnection.videoOrientation = orientation
            }
            
            if previewConnection.isVideoMirroringSupported {
                previewConnection.automaticallyAdjustsVideoMirroring = false
                // Only update mirroring if it changed
                if previewConnection.isVideoMirrored != shouldMirror {
                    previewConnection.isVideoMirrored = shouldMirror
                    print("UVC: Preview layer mirroring updated: \(shouldMirror)")
                }
            }
        }
        
        print("UVC: Video orientation updated to \(orientation.rawValue)")
    }
    
    @MainActor
    func integrateWithYOLOView(view: UIView) {
        if let previewLayer = self.previewLayer {
            view.layer.insertSublayer(previewLayer, at: 0)
            previewLayer.frame = view.bounds
            
            // Set appropriate video gravity based on orientation
            let orientation = UIDevice.current.orientation
            setVideoGravityForOrientation(orientation: orientation)
            
            // Ensure proper orientation setup
            updateForOrientationChange(orientation: orientation)
            
            print("UVC: Integrated with YOLOView - frame: \(previewLayer.frame), gravity: \(previewLayer.videoGravity.rawValue)")
        }
    }
    
    @MainActor
    func addOverlayLayer(_ layer: CALayer) {
        if let previewLayer = self.previewLayer {
            previewLayer.addSublayer(layer)
        }
    }
    
    @MainActor
    func addBoundingBoxViews(_ boxViews: [BoundingBoxView]) {
        if let previewLayer = self.previewLayer {
            for box in boxViews {
                box.addToLayer(previewLayer)
            }
        }
    }
    
    // MARK: - Coordinate Transformation (same as CameraVideoSource)
    
    @MainActor
    func transformDetectionToScreenCoordinates(
        rect: CGRect,
        viewBounds: CGRect,
        orientation: UIDeviceOrientation
    ) -> CGRect {
        let width = viewBounds.width
        let height = viewBounds.height
        
        // Use last known orientation for flat/unknown device orientations
        let effectiveOrientation: UIDeviceOrientation
        switch orientation {
        case .faceUp, .faceDown, .unknown:
            effectiveOrientation = lastKnownOrientation
            // Reduce logging noise - only log occasionally for coordinate transforms with flat orientations
            if frameProcessingCount % 900 == 0 { // Log every ~30 seconds at 30fps
                print("UVC: Coordinate transform using last known orientation \(lastKnownOrientation.rawValue) (device: \(orientation.rawValue))")
            }
        default:
            effectiveOrientation = orientation
        }
        
        if effectiveOrientation == .portrait || effectiveOrientation == .portraitUpsideDown {
            // For UVC cameras in portrait mode with letterboxing (resizeAspect)
            // The stream maintains 16:9 aspect ratio in the center with black bars
            let frameAspectRatio: CGFloat = 16.0 / 9.0  // UVC stream is 1280×720
            let viewAspectRatio = height / width
            
            // Calculate the actual display area of the video stream (excluding black bars)
            let streamHeight = width / frameAspectRatio  // Height of the video stream area
            let blackBarHeight = (height - streamHeight) / 2  // Height of each black bar
            
            // Scale coordinates to the actual stream area
            let scaledY = rect.origin.y * streamHeight + blackBarHeight
            let scaledHeight = rect.size.height * streamHeight
            
            // Apply coordinate system flip for portrait
            let flippedRect = CGRect(
                x: rect.origin.x * width,
                y: height - scaledY - scaledHeight,
                width: rect.size.width * width,
                height: scaledHeight
            )
            
            return flippedRect
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
            
            let screenRect = CGRect(
                x: rect.origin.x * longSide * scaleX - offsetX,
                y: height - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY),
                width: rect.size.width * longSide * scaleX,
                height: rect.size.height * shortSide * scaleY
            )
            
            return screenRect
        }
    }
    
    // MARK: - Performance Analysis
    
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
    func captureOutput(
        _ output: AVCaptureOutput, 
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
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
