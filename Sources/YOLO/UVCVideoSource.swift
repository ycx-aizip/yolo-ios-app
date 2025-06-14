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
    
    // MARK: - UVC Configuration
    
    /// Determines the optimal session preset for UVC cameras with adaptive fallback
    @MainActor
    private var optimalSessionPreset: AVCaptureSession.Preset {
        guard let device = captureDevice else { return .medium }
        
        // Priority: HD 720p → HD 1080p → Medium fallback
        if device.supportsSessionPreset(.hd1280x720) {
            return .hd1280x720
        } else if device.supportsSessionPreset(.hd1920x1080) {
            return .hd1920x1080
        } else {
            return .medium
        }
    }
    
    /// Human-readable description of the current session preset
    @MainActor
    private var sessionPresetDescription: String {
        switch optimalSessionPreset {
        case .hd1280x720: return "HD 1280×720 (UVC Optimized)"
        case .hd1920x1080: return "Full HD 1920×1080 (UVC)"
        case .medium: return "Medium Quality (UVC Fallback)"
        default: return "UVC Default"
        }
    }
    
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

    // MARK: - UVC Device Discovery
    
    /// Discovers available UVC (external) cameras with basic validation
    /// - Returns: Array of available external camera devices
    @MainActor
    static func discoverUVCCameras() -> [AVCaptureDevice] {
        print("UVC: Starting device discovery...")
        
        guard #available(iOS 17.0, *) else {
            print("UVC: External cameras require iOS 17.0 or later")
            return []
        }
        
        let discoverySession = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external],
            mediaType: .video,
            position: .unspecified
        )
        
        let allDevices = discoverySession.devices
        print("UVC: Found \(allDevices.count) total devices in discovery session")
        
        let filteredDevices = allDevices.filter { device in
            let isValid = device.hasMediaType(.video) && device.isConnected && device.deviceType == .external
            print("UVC: Device \(device.localizedName): \(isValid ? "VALID" : "INVALID")")
            return isValid
        }
        
        print("UVC: Discovery complete - \(filteredDevices.count) suitable external cameras")
        return filteredDevices
    }
    
    /// Gets the best available UVC camera device
    /// - Returns: The first available external camera, or nil if none found
    @MainActor
    static func bestUVCDevice() -> AVCaptureDevice? {
        let uvcCameras = discoverUVCCameras()
        
        print("UVC: Found \(uvcCameras.count) external camera(s)")
        for (index, camera) in uvcCameras.enumerated() {
            print("UVC: Device \(index): \(camera.localizedName)")
        }
        
        let bestDevice = uvcCameras.first
        if let device = bestDevice {
            print("UVC: Selected best device: \(device.localizedName) (Model: \(device.modelID ?? "Unknown"))")
        } else {
            print("UVC: No external cameras found")
        }
        
        return bestDevice
    }

    // MARK: - FrameSource Protocol Implementation
    
    /// Sets up the UVC camera with specified configuration
    /// - Parameter completion: Called when setup is complete, with a Boolean indicating success
    @MainActor
    func setUp(completion: @escaping @Sendable (Bool) -> Void) {
        print("UVC: Starting setup...")
        
        guard let uvcDevice = Self.bestUVCDevice() else {
            print("UVC: No external cameras found during setup")
            completion(false)
            return
        }
        
        self.captureDevice = uvcDevice
        print("UVC: Selected device for setup: \(uvcDevice.localizedName)")
        
        let preset = self.optimalSessionPreset
        let orientation = UIDevice.current.orientation
        
        Task.detached {
            let success = await self.setUpUVCCamera(sessionPreset: preset, orientation: orientation)
            await MainActor.run {
                print("UVC: Setup completed with success: \(success)")
                completion(success)
            }
        }
    }
    
    /// Sets up the UVC camera with detailed configuration
    /// - Parameters:
    ///   - sessionPreset: The capture session preset to use
    ///   - orientation: The current device orientation
    /// - Returns: True if setup was successful, false otherwise
    func setUpUVCCamera(sessionPreset: AVCaptureSession.Preset, orientation: UIDeviceOrientation) async -> Bool {
        print("UVC: Beginning camera configuration...")
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset
        
        guard let captureDevice = captureDevice else {
            print("UVC: No capture device available")
            captureSession.commitConfiguration()
            return false
        }
        
        await MainActor.run {
            let deviceType = UIDevice.current.userInterfaceIdiom == .pad ? "iPad" : "iPhone"
            let presetDescription = self.sessionPresetDescription
            print("UVC: Configuring \(deviceType) with \(presetDescription)")
        }

        // Create device input
        do {
            print("UVC: Creating device input for \(captureDevice.localizedName)")
            videoInput = try AVCaptureDeviceInput(device: captureDevice)
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
        print("UVC: Video input added to session successfully")
        
        // Configure preview layer with consistent orientation
        print("UVC: Creating preview layer...")
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = orientation.isPortrait ? .resizeAspect : .resizeAspectFill
        previewLayer.connection?.videoOrientation = .landscapeRight
        self.previewLayer = previewLayer
        print("UVC: Preview layer created with gravity: \(previewLayer.videoGravity.rawValue)")

        // Configure video output
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]
        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: cameraQueue)
        
        guard captureSession.canAddOutput(videoOutput) else {
            print("UVC: Cannot add video output to session")
            captureSession.commitConfiguration()
            return false
        }
        captureSession.addOutput(videoOutput)

        // Configure video connection with consistent orientation and no mirroring
        let connection = videoOutput.connection(with: AVMediaType.video)
        connection?.videoOrientation = .landscapeRight
        
        if let conn = connection, conn.isVideoMirroringSupported {
            conn.automaticallyAdjustsVideoMirroring = false
            conn.isVideoMirrored = false
            print("UVC: Video output mirroring configured: false")
        }
        
        if let previewConn = previewLayer.connection, previewConn.isVideoMirroringSupported {
            previewConn.automaticallyAdjustsVideoMirroring = false
            previewConn.isVideoMirrored = false
            print("UVC: Preview layer mirroring configured: false")
        }

        // Configure device settings if supported
        do {
            try captureDevice.lockForConfiguration()
            
            if captureDevice.isFocusModeSupported(.continuousAutoFocus) &&
               captureDevice.isFocusPointOfInterestSupported {
                captureDevice.focusMode = .continuousAutoFocus
                captureDevice.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            }
            
            if captureDevice.isExposureModeSupported(.continuousAutoExposure) {
                captureDevice.exposureMode = .continuousAutoExposure
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
    
    /// Shows UI for selecting UVC content (not applicable)
    /// - Parameters:
    ///   - viewController: The view controller to present UI from
    ///   - completion: Called when selection is complete with true
    @MainActor
    func showContentSelectionUI(from viewController: UIViewController, completion: @escaping (Bool) -> Void) {
        completion(true)
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
    
    /// Integrates the UVC source with a YOLOView for proper display
    /// - Parameter view: The YOLOView to integrate with
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
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
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
