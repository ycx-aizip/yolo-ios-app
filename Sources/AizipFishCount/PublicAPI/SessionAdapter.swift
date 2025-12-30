//
//  SessionAdapter.swift
//  AizipFishCount
//
//  Internal adapter that wraps TrackingDetector to implement FishCountSession
//  This class bridges the internal implementation with the public API
//

import Foundation
import CoreGraphics
import CoreVideo
import Vision
import AVFoundation

// MARK: - Helper Classes

/// Internal listener wrapper for ResultsListener protocol
private class DetectorResultsListener: ResultsListener {
    private let callback: (YOLOResult) -> Void

    init(callback: @escaping (YOLOResult) -> Void) {
        self.callback = callback
    }

    func on(result: YOLOResult) {
        callback(result)
    }
}

// MARK: - Session Adapter Implementation

/// Internal adapter that implements FishCountSession by wrapping TrackingDetector
@MainActor
internal class SessionAdapter: FishCountSession {

    // MARK: - Private Properties

    /// The internal tracking detector instance (mutable for model switching)
    private var detector: TrackingDetector

    /// Model name
    private let modelName: String

    /// Weak reference to delegate
    private weak var delegate: FishCountSessionDelegate?

    /// Last known count (for change detection)
    private var lastCount: Int = 0

    /// Current counting direction
    private var currentDirection: CountingDirection

    /// Listener reference to keep it alive
    private var resultsListener: DetectorResultsListener?

    /// Pause state
    private var isPausedState: Bool = false

    /// Current model name (mutable for switching)
    private var currentModelName: String

    // MARK: - Initialization

    /// Initialize the session adapter
    ///
    /// - Parameters:
    ///   - modelName: Name of the CoreML model to use
    ///   - detector: The TrackingDetector instance (injected for testability)
    ///   - delegate: Optional delegate for callbacks
    internal init(modelName: String, detector: TrackingDetector, delegate: FishCountSessionDelegate? = nil) {
        self.modelName = modelName
        self.currentModelName = modelName
        self.detector = detector
        self.delegate = delegate
        self.currentDirection = ThresholdCounter.defaultCountingDirection

        // Apply shared configuration to detector
        detector.applySharedConfiguration()

        // Set up callbacks from detector to delegate
        setupDetectorCallbacks()
    }

    // MARK: - Detector Callbacks Setup

    /// Configure callbacks from TrackingDetector to forward to delegate
    private func setupDetectorCallbacks() {
        // Create a results listener wrapper
        let resultsListener = DetectorResultsListener { [weak self] result in
            guard let self = self else { return }

            // Convert internal YOLOResult to public DetectionResult
            let detectionResult = self.convertToDetectionResult(result)
            self.delegate?.session(self, didDetect: detectionResult)

            // Check for count changes
            let currentCount = self.detector.getCount()
            if currentCount != self.lastCount {
                self.lastCount = currentCount
                self.delegate?.session(self, countDidChange: currentCount)
            }
        }

        // Store the listener reference to keep it alive
        self.resultsListener = resultsListener
        self.detector.currentOnResultsListener = resultsListener

        // Forward calibration progress
        detector.onCalibrationProgress = { [weak self] current, total in
            guard let self = self else { return }
            self.delegate?.session(self, calibrationProgress: current, total: total)
        }

        // Forward calibration completion
        detector.onCalibrationSummary = { [weak self] summary in
            guard let self = self else { return }

            // Convert internal CalibrationSummary to public CalibrationResult
            let result = CalibrationResult(
                thresholds: summary.thresholds,
                detectedDirection: summary.detectedDirection,
                originalDirection: summary.originalDirection,
                confidence: summary.movementAnalysisSuccess ? 1.0 : 0.0,
                warnings: summary.warnings,
                thresholdCalibrationEnabled: summary.thresholdCalibrationEnabled,
                directionCalibrationEnabled: summary.directionCalibrationEnabled
            )

            self.delegate?.session(self, calibrationDidComplete: result)
        }
    }

    // MARK: - Result Conversion

    /// Convert internal YOLOResult to public DetectionResult
    private func convertToDetectionResult(_ yoloResult: YOLOResult) -> DetectionResult {
        var detectionBoxes: [DetectionBox] = []

        for box in yoloResult.boxes {
            // Get tracking info from detector
            let trackInfo = detector.getTrackInfo(for: box)

            let detectionBox = DetectionBox(
                trackId: trackInfo?.trackId,
                boundingBox: box.xywhn,
                confidence: box.conf,
                label: box.cls,
                isCounted: trackInfo?.isCounted ?? false
            )

            detectionBoxes.append(detectionBox)
        }

        return DetectionResult(
            boxes: detectionBoxes,
            fps: yoloResult.fps ?? 0.0,
            inferenceTime: (yoloResult.speed ?? 0.0) * 1000, // Convert to milliseconds
            timestamp: CACurrentMediaTime()
        )
    }

    // MARK: - FishCountSession Protocol Implementation

    public func processFrame(_ pixelBuffer: CVPixelBuffer) {
        // Skip processing if paused
        guard !isPausedState else { return }

        // First, let TrackingDetector process for calibration if needed
        detector.processFrame(pixelBuffer)

        // Convert CVPixelBuffer to CMSampleBuffer for Vision API
        guard let sampleBuffer = createSampleBuffer(from: pixelBuffer) else {
            let error = FishCountError.processingFailed("Failed to convert pixel buffer to sample buffer")
            delegate?.session(self, didFailWithError: error)
            return
        }

        // Trigger YOLO inference through the detector
        // The detector's predict method will call processObservations, which triggers our callbacks
        detector.predict(sampleBuffer: sampleBuffer, onResultsListener: nil, onInferenceTime: nil)
    }

    /// Helper to convert CVPixelBuffer to CMSampleBuffer
    private func createSampleBuffer(from pixelBuffer: CVPixelBuffer) -> CMSampleBuffer? {
        var sampleBuffer: CMSampleBuffer?

        var timingInfo = CMSampleTimingInfo()
        timingInfo.presentationTimeStamp = CMTime(seconds: CACurrentMediaTime(), preferredTimescale: 600)
        timingInfo.duration = CMTime.invalid
        timingInfo.decodeTimeStamp = CMTime.invalid

        var formatDescription: CMFormatDescription?
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDescription
        )

        guard let formatDesc = formatDescription else {
            return nil
        }

        let status = CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDesc,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )

        guard status == noErr else {
            return nil
        }

        return sampleBuffer
    }

    public func setThresholds(_ values: [CGFloat]) {
        guard !values.isEmpty else { return }

        // Convert display coordinates to counting coordinates
        let countingThresholds = UnifiedCoordinateSystem.displayToCounting(values, countingDirection: currentDirection)

        // Set thresholds with both counting and display values
        detector.setThresholds(countingThresholds, originalDisplayValues: values)
    }

    public func setCountingDirection(_ direction: CountingDirection) {
        currentDirection = direction
        detector.setCountingDirection(direction)
    }

    public func getCount() -> Int {
        return detector.getCount()
    }

    public func resetCount() {
        detector.resetCount()
        lastCount = 0
        delegate?.session(self, countDidChange: 0)
    }

    public func getThresholds() -> [CGFloat] {
        let countingThresholds = detector.getThresholds()
        // Convert counting coordinates back to display coordinates
        return UnifiedCoordinateSystem.countingToDisplay(countingThresholds, countingDirection: currentDirection)
    }

    public func getCountingDirection() -> CountingDirection {
        return currentDirection
    }

    public func getDetectionConfig() -> DetectionConfig {
        return DetectionConfig(
            confidenceThreshold: Float(detector.confidenceThreshold ?? 0.25),
            iouThreshold: Float(detector.iouThreshold ?? 0.80),
            maxDetections: detector.numItemsThreshold
        )
    }

    public func updateDetectionConfig(_ config: DetectionConfig) {
        detector.confidenceThreshold = Double(config.confidenceThreshold)
        detector.iouThreshold = Double(config.iouThreshold)
        detector.numItemsThreshold = config.maxDetections
    }

    public func setAutoCalibration(enabled: Bool, config: CalibrationConfig?) {
        // Apply calibration config if provided
        if let config = config {
            AutoCalibrationConfig.shared.setConfiguration(
                thresholdCalibration: config.enableThresholdCalibration,
                directionCalibration: config.enableDirectionCalibration,
                thresholdFrames: config.thresholdFrames,
                movementFrames: config.movementFrames
            )
        }

        // Enable/disable calibration on detector
        detector.setAutoCalibration(enabled: enabled)
    }

    public func isCalibrating() -> Bool {
        return detector.getAutoCalibrationEnabled()
    }

    public func getCalibrationProgress() -> (current: Int, total: Int)? {
        guard isCalibrating() else { return nil }

        let current = detector.getCalibrationFrameCount()
        let total = detector.getTargetCalibrationFrames()

        return (current, total)
    }

    public func getModelName() -> String {
        return currentModelName
    }

    public func setDelegate(_ delegate: FishCountSessionDelegate?) {
        self.delegate = delegate
    }

    public func switchModel(to modelName: String, completion: ((Result<Void, FishCountError>) -> Void)?) {
        // Find the new model in the bundle
        guard let modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") ??
                             Bundle.main.url(forResource: modelName, withExtension: "mlmodel") else {
            let error = FishCountError.modelNotFound(modelName)
            completion?(.failure(error))
            return
        }

        // Save current session state
        let currentThresholds = getThresholds()
        let currentDirection = getCountingDirection()
        let currentCount = getCount()
        let currentConfig = getDetectionConfig()

        // Load new model asynchronously
        TrackingDetector.create(unwrappedModelURL: modelURL, isRealTime: true) { [weak self] result in
            guard let self = self else { return }

            Task { @MainActor in
                switch result {
                case .success(let newDetector):
                    guard let trackingDetector = newDetector as? TrackingDetector else {
                        let error = FishCountError.processingFailed("Failed to create tracking detector")
                        completion?(.failure(error))
                        return
                    }

                    // Apply shared configuration
                    trackingDetector.applySharedConfiguration()

                    // Restore session state
                    trackingDetector.confidenceThreshold = Double(currentConfig.confidenceThreshold)
                    trackingDetector.iouThreshold = Double(currentConfig.iouThreshold)
                    trackingDetector.numItemsThreshold = currentConfig.maxDetections
                    trackingDetector.setCountingDirection(currentDirection)

                    // Convert display thresholds to counting coordinates
                    let countingThresholds = UnifiedCoordinateSystem.displayToCounting(currentThresholds, countingDirection: currentDirection)
                    trackingDetector.setThresholds(countingThresholds, originalDisplayValues: currentThresholds)

                    // NOTE: Count cannot be restored after model switch
                    // The new detector starts with count=0 because track IDs are reset
                    // This is acceptable behavior - model switch implies a fresh start

                    // Update detector reference
                    self.detector = trackingDetector

                    // Update model name
                    self.currentModelName = modelName

                    // Re-setup callbacks for the new detector
                    self.setupDetectorCallbacks()

                    // Notify delegate that count was reset to 0
                    self.lastCount = 0
                    self.delegate?.session(self, countDidChange: 0)

                    completion?(.success(()))

                case .failure(let error):
                    let fishCountError = FishCountError.modelNotFound("\(modelName): \(error.localizedDescription)")
                    completion?(.failure(fishCountError))
                }
            }
        }
    }

    public func pause() {
        isPausedState = true
    }

    public func resume() {
        isPausedState = false
    }

    public func isPaused() -> Bool {
        return isPausedState
    }
}
