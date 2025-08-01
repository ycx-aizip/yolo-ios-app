// from Aizip

//  The TrackingDetector class extends ObjectDetector with tracking capabilities using ByteTrack.
//  It tracks objects across frames, associates detections with existing tracks, and supports
//  threshold crossing detection for counting applications like fish counting. The implementation
//  maintains tracking state for each detected object including position history and counting status.

import Foundation
import UIKit
import Vision

/// Direction of fish movement
/// These represent the actual movement direction of the fish:
/// - up: Fish moving upward (used with CountingDirection.bottomToTop)
/// - down: Fish moving downward (used with CountingDirection.topToBottom)
/// - left: Fish moving leftward (used with CountingDirection.rightToLeft)
/// - right: Fish moving rightward (used with CountingDirection.leftToRight)
private enum Direction {
    case up
    case down
    case left
    case right
}

/// Centralized configuration for all TrackingDetector instances
/// This ensures consistent initial setup across all frame sources
@MainActor
public class TrackingDetectorConfig {
    /// Shared singleton instance
    public static let shared = TrackingDetectorConfig()
    
    /// Default thresholds for counting (normalized coordinates, 0.0-1.0)
    public var defaultThresholds: [CGFloat] = [0.2, 0.4]
    
    /// Default counting direction
    public var defaultCountingDirection: CountingDirection = .bottomToTop
    
    /// Default confidence threshold for YOLO detection
    public var defaultConfidenceThreshold: Float = 0.60
    
    /// Default IoU threshold for non-maximum suppression
    public var defaultIoUThreshold: Float = 0.50
    
    /// Default number of items threshold for detection display
    public var defaultNumItemsThreshold: Int = 100
    
    private init() {
        // Private initializer to enforce singleton pattern
    }
    
    /// Update all default values at once
    public func updateDefaults(
        thresholds: [CGFloat]? = nil,
        countingDirection: CountingDirection? = nil,
        confidenceThreshold: Float? = nil,
        iouThreshold: Float? = nil,
        numItemsThreshold: Int? = nil
    ) {
        if let thresholds = thresholds {
            self.defaultThresholds = thresholds
        }
        if let countingDirection = countingDirection {
            self.defaultCountingDirection = countingDirection
        }
        if let confidenceThreshold = confidenceThreshold {
            self.defaultConfidenceThreshold = confidenceThreshold
        }
        if let iouThreshold = iouThreshold {
            self.defaultIoUThreshold = iouThreshold
        }
        if let numItemsThreshold = numItemsThreshold {
            self.defaultNumItemsThreshold = numItemsThreshold
        }
        
        print("TrackingDetector: Configuration updated")
    }
}

/**
 * TrackingDetector
 *
 * Maps to Python Implementation:
 * - Primary Correspondence: The main fish counting logic in `counting_demo2.py`
 * - Core functionality:
 *   - Integrates with ByteTracker for object tracking
 *   - Implements threshold crossing detection logic
 *   - Maintains count of objects that cross thresholds
 *   - Updates tracking state for visualization
 *
 * Implementation Details:
 * - Extends ObjectDetector class to maintain compatibility with YOLO app
 * - Implements threshold crossing logic similar to `check_threshold_crossing` in Python
 * - Maintains tracking states similar to `track_states` dictionary in Python
 * - Provides methods to integrate tracking information with visualization
 */
class TrackingDetector: ObjectDetector {
    
    /// The ByteTracker instance used for tracking objects
    @MainActor private lazy var byteTracker = ByteTracker()
    
    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0
    
    /// Thresholds used for counting (normalized coordinates, 0.0-1.0)
    /// For vertical directions (topToBottom, bottomToTop), these are y-coordinates
    /// For horizontal directions (leftToRight, rightToLeft), these are x-coordinates
    private var thresholds: [CGFloat]
    
    /// Map of track IDs to counting status
    private var countedTracks: [Int: Bool] = [:]
    
    /// The current tracked objects
    private var trackedObjects: [STrack] = []
    
    /// Current counting direction
    private var countingDirection: CountingDirection
    
    /// Direction of fish movement
    private var crossingDirections: [Int: Direction] = [:]
    
    /// Previous positions for each track
    private var previousPositions: [Int: (x: CGFloat, y: CGFloat)] = [:]
    
    /// Map of track positions from 5 frames ago for detecting fast movements
    private var historyPositions: [Int: (x: CGFloat, y: CGFloat)] = [:]
    
    /// Frame counter for less frequent cleanup
    private var frameCount: Int = 0
    
    // MARK: - Auto-calibration properties
    
    /// Flag indicating if auto-calibration is enabled
    private var isAutoCalibrationEnabled: Bool = false
    
    /// Flag indicating if calibration has been completed
    private var isCalibrated: Bool = false
    
    /// Current calibration phase
    enum CalibrationPhase {
        case thresholdDetection    // Phase 1: OpenCV edge detection
        case movementAnalysis      // Phase 2: YOLO + movement analysis
        case completed            // Calibration finished
    }
    
    private var calibrationPhase: CalibrationPhase = .thresholdDetection
    
    /// Flag to track if Phase 1 was actually executed (vs bypassed)
    private var wasPhase1Executed: Bool = false
    
    /// Original display thresholds before calibration (for bypass mode)
    private var originalDisplayThresholds: [CGFloat] = []
    
    /// Current frame number in calibration sequence
    private var calibrationFrameCount: Int = 0
    
    /// Total number of frames to process for calibration
    private var targetCalibrationFrames: Int = 300
    
    /// Reference to the current frame being processed
    private var currentPixelBuffer: CVPixelBuffer?
    
    // MARK: - Phase 2: Movement Analysis Properties
    
    /// Movement analysis frame count (separate from threshold calibration)
    private var movementAnalysisFrameCount: Int = 0
    
    /// Fish movement data collection for direction analysis
    private var fishMovementData: [Int: FishMovementData] = [:]
    
    /// Original counting direction before auto-detection
    private var originalCountingDirection: CountingDirection
    
    /// Flag to prevent counting during movement analysis
    private var isMovementAnalysisPhase: Bool = false
    
    // MARK: - Unified Calibration Callbacks
    
    /// Callback for reporting calibration progress (unified for both phases)
    var onCalibrationProgress: ((Int, Int) -> Void)?
    
    /// Callback for reporting calibration completion with new thresholds
    var onCalibrationComplete: (([CGFloat]) -> Void)?
    
    /// Callback for reporting direction detection
    var onDirectionDetected: ((CountingDirection) -> Void)?
    
    /// Callback for reporting complete calibration summary
    var onCalibrationSummary: ((CalibrationSummary) -> Void)?
    
    // MARK: - Initialization
    
    /// Initialize TrackingDetector with default values
    /// Configuration will be applied after initialization via applySharedConfiguration()
    required init() {
        // Initialize with hardcoded defaults (will be overridden by shared config)
        self.thresholds = [0.2, 0.4]
        self.countingDirection = .bottomToTop
        self.originalCountingDirection = .bottomToTop
        
        super.init()
    }
    
    /// Apply the shared configuration to this instance
    /// This must be called after initialization to apply the centralized settings
    @MainActor
    func applySharedConfiguration() {
        let config = TrackingDetectorConfig.shared
        
        // Apply configuration values
        self.thresholds = config.defaultThresholds
        self.countingDirection = config.defaultCountingDirection
        self.confidenceThreshold = Double(config.defaultConfidenceThreshold)
        self.iouThreshold = Double(config.defaultIoUThreshold)
        self.numItemsThreshold = config.defaultNumItemsThreshold
        
        // Set the expected movement direction in STrack
        STrack.expectedMovementDirection = self.countingDirection
        
        // Update tracking parameters based on the direction
        TrackingParameters.updateParametersForCountingDirection(self.countingDirection)
    }
    
    // MARK: - Threshold management
    
    /// Sets the thresholds for counting
    @MainActor
    func setThresholds(_ values: [CGFloat]) {
        guard values.count >= 1 else { return }
        
        // Ensure thresholds are within valid range (0.0-1.0)
        let validThresholds = values.map { max(0.0, min(1.0, $0)) }
        self.thresholds = validThresholds
    }
    
    /// Sets the thresholds with original display values (for bypass mode)
    @MainActor
    func setThresholds(_ countingValues: [CGFloat], originalDisplayValues: [CGFloat]) {
        guard countingValues.count >= 1 && originalDisplayValues.count >= 1 else { return }
        
        // Store counting thresholds for internal use
        let validCountingThresholds = countingValues.map { max(0.0, min(1.0, $0)) }
        self.thresholds = validCountingThresholds
        
        // Store original display thresholds for bypass mode
        let validDisplayThresholds = originalDisplayValues.map { max(0.0, min(1.0, $0)) }
        self.originalDisplayThresholds = validDisplayThresholds
    }
    
    /// Gets the current count of objects that have crossed the threshold
    ///
    /// - Returns: The total count
    func getCount() -> Int {
        return totalCount
    }
    
    /// Resets the counting state and clears all tracked objects
    @MainActor
    public func resetCount() {
        totalCount = 0
        
        // Clean up any tracked objects to release memory
        for track in trackedObjects {
            track.cleanup()
        }
        
        countedTracks.removeAll(keepingCapacity: true)
        crossingDirections.removeAll(keepingCapacity: true)
        previousPositions.removeAll(keepingCapacity: true)
        historyPositions.removeAll(keepingCapacity: true)
        frameCount = 0
        
        // Reset the ByteTracker to reset the track IDs
        byteTracker.reset()
        trackedObjects.removeAll(keepingCapacity: true)
    }
    
    /// Sets the counting direction
    @MainActor
    func setCountingDirection(_ direction: CountingDirection) {
        self.countingDirection = direction
        
        // Update the expected movement direction in STrack
        STrack.expectedMovementDirection = direction
        
        // Update tracking parameters based on the new direction
        TrackingParameters.updateParametersForCountingDirection(direction)
        
        // Reset calibration state when changing direction
        resetCalibration()
    }
    
    // MARK: - Auto-calibration methods
    
    /// Enable or disable auto-calibration
    @MainActor
    func setAutoCalibration(enabled: Bool) {
        // If we're disabling calibration that was previously enabled, clean up properly
        if !enabled && isAutoCalibrationEnabled {
            resetCalibration()
        }
        
        // Set the new state
        isAutoCalibrationEnabled = enabled
        
        if enabled {
            let config = AutoCalibrationConfig.shared
            
            // Store original direction for comparison
            originalCountingDirection = countingDirection
            
            // Reset calibration state when enabling
            resetCalibration()
            
            // Initialize calibration state
            calibrationFrameCount = 0
            movementAnalysisFrameCount = 0
            fishMovementData.removeAll(keepingCapacity: true)
            isMovementAnalysisPhase = false
            wasPhase1Executed = false  // Reset the Phase 1 execution flag
            // Note: originalDisplayThresholds are already set by setThresholds() call
            
            // Determine starting phase based on configuration
            if config.isThresholdCalibrationEnabled {
                calibrationPhase = .thresholdDetection
                targetCalibrationFrames = config.thresholdCalibrationFrames
                print("AutoCalibration: Starting Phase 1 - Threshold Detection (\(config.thresholdCalibrationFrames) frames)")
            } else if config.isDirectionCalibrationEnabled {
                // Even when Phase 1 is disabled, we start with threshold detection for bypass
                calibrationPhase = .thresholdDetection
                targetCalibrationFrames = 2  // Quick bypass for Phase 1
                print("AutoCalibration: Phase 1 disabled - Quick bypass (2 frames) then Phase 2")
            } else {
                print("AutoCalibration: ERROR - No phases configured")
                isAutoCalibrationEnabled = false
                return
            }
            
            // Clear all tracked objects to ensure clean start for calibration
            trackedObjects.removeAll(keepingCapacity: true)
            countedTracks.removeAll(keepingCapacity: true)
            crossingDirections.removeAll(keepingCapacity: true)
            previousPositions.removeAll(keepingCapacity: true)
            historyPositions.removeAll(keepingCapacity: true)
            
            // Reset the ByteTracker to clear any existing tracks
            byteTracker.reset()
            
            print("AutoCalibration: Started - Total frames needed: \(config.totalCalibrationFrames)")
        }
    }
    
    /// Get the current calibration frame count
    @MainActor
    func getCalibrationFrameCount() -> Int {
        return calibrationFrameCount
    }
    
    /// Get the current calibration phase for frame source routing
    @MainActor
    func getCalibrationPhase() -> CalibrationPhase {
        return calibrationPhase
    }
    
    /// Reset calibration state
    @MainActor
    private func resetCalibration() {
        isCalibrated = false
        calibrationFrameCount = 0
        movementAnalysisFrameCount = 0
        calibrationPhase = .thresholdDetection
        isMovementAnalysisPhase = false
        fishMovementData.removeAll(keepingCapacity: true)
        currentPixelBuffer = nil
    }
    
    /// Process a frame for calibration (streaming approach)
    @MainActor
    private func processFrameForCalibration(_ pixelBuffer: CVPixelBuffer) {
        // Skip if calibration is completed
        if isCalibrated {
            return
        }
        
        calibrationFrameCount += 1
        let config = AutoCalibrationConfig.shared
        
        // Calculate total progress across all enabled phases
        let totalFrames = config.totalCalibrationFrames
        let currentTotalFrames = calibrationFrameCount + movementAnalysisFrameCount
        
        // Report unified progress via callback
        onCalibrationProgress?(currentTotalFrames, totalFrames)
        
        // Check if threshold calibration is disabled - use bypass mode
        if !config.isThresholdCalibrationEnabled {
            // Bypass mode: Keep current thresholds and complete quickly (in 2 frames for UI consistency)
            if calibrationFrameCount >= 2 {
                // DO NOT SORT - keep current thresholds exactly as they are
                // The user has manually set these values and we should preserve them
                wasPhase1Executed = false  // Mark as bypassed
                
                print("AutoCalibration: Phase 1 bypassed - Using current thresholds: \(thresholds)")
                completeThresholdCalibration()
            }
            return
        }
        
        // Normal threshold calibration mode
        // Mark that Phase 1 is actually being executed
        wasPhase1Executed = true
        
        // Process frame with OpenCV directly for calibration
        let isVerticalDirection = countingDirection == .topToBottom || countingDirection == .bottomToTop
        
        if let thresholds = OpenCVWrapper.processCalibrationFrame(pixelBuffer, isVerticalDirection: isVerticalDirection) as? [NSNumber], 
           thresholds.count >= 2 {
            // Accumulate these threshold values (we'll average them at the end)
            let threshold1 = CGFloat(thresholds[0].floatValue)
            let threshold2 = CGFloat(thresholds[1].floatValue)
            
            // Update our running thresholds
            if calibrationFrameCount == 1 {
                // First frame - just use the values directly
                self.thresholds = [threshold1, threshold2]
            } else {
                // Accumulate by blending with previous values using weighted average
                // Give more weight to newer frames using a simple exponential moving average
                let weight = 2.0 / Double(calibrationFrameCount + 1) 
                let newThreshold1 = CGFloat(weight) * threshold1 + CGFloat(1 - weight) * self.thresholds[0]
                let newThreshold2 = CGFloat(weight) * threshold2 + CGFloat(1 - weight) * self.thresholds[1]
                self.thresholds = [newThreshold1, newThreshold2]
            }
        }
        
        // Check if we've processed enough frames for this phase
        if calibrationFrameCount >= config.thresholdCalibrationFrames {
            completeThresholdCalibration()
        }
    }
    
    /// Complete the threshold calibration phase (Phase 1)
    @MainActor
    private func completeThresholdCalibration() {
        let config = AutoCalibrationConfig.shared
        
        // Only sort thresholds if Phase 1 was actually executed (not bypassed)
        if wasPhase1Executed {
            // Sort thresholds to ensure proper order for OpenCV-detected values
            let sortedThresholds = thresholds.sorted()
            self.thresholds = sortedThresholds
            print("AutoCalibration: Phase 1 complete (executed) - Thresholds sorted: \(thresholds)")
        } else {
            // Phase 1 was bypassed - keep current thresholds exactly as set by user
            print("AutoCalibration: Phase 1 complete (bypassed) - Thresholds preserved: \(thresholds)")
        }
        
        // Notify threshold completion
        onCalibrationComplete?(thresholds)
        
        // Check what to do next
        if config.isDirectionCalibrationEnabled {
            // Start Phase 2 immediately and sequentially
            print("AutoCalibration: Starting Phase 2 immediately after Phase 1")
            startMovementAnalysisPhaseSequentially()
        } else {
            // No Phase 2 needed, complete immediately
            print("AutoCalibration: Phase 2 not enabled, completing calibration")
            completeEntireCalibration()
        }
    }
    
    /// Start the movement analysis phase (Phase 2) - SEQUENTIAL VERSION
    @MainActor
    private func startMovementAnalysisPhaseSequentially() {
        let config = AutoCalibrationConfig.shared
        
        // Step 1: Update phase state immediately
        calibrationPhase = .movementAnalysis
        movementAnalysisFrameCount = 0
        isMovementAnalysisPhase = true
        
        // Step 2: Clear all tracking data synchronously
        fishMovementData.removeAll(keepingCapacity: true)
        trackedObjects.removeAll(keepingCapacity: true)
        countedTracks.removeAll(keepingCapacity: true)
        crossingDirections.removeAll(keepingCapacity: true)
        previousPositions.removeAll(keepingCapacity: true)
        historyPositions.removeAll(keepingCapacity: true)
        
        // Step 3: Reset tracker synchronously
        byteTracker.reset()
        
        print("AutoCalibration: Phase 2 ready - Movement Analysis (\(config.movementAnalysisFrames) frames)")
    }
    
    /// Process movement analysis for Phase 2
    @MainActor
    private func processMovementAnalysis() {
        // Safety checks to prevent re-entry and stuck states
        guard isMovementAnalysisPhase && 
              calibrationPhase == .movementAnalysis &&
              isAutoCalibrationEnabled else { 
            return 
        }
        
        let config = AutoCalibrationConfig.shared
        movementAnalysisFrameCount += 1
        
        // Safety mechanism: prevent infinite analysis if something goes wrong
        if movementAnalysisFrameCount > config.movementAnalysisFrames + 100 {
            print("AutoCalibration: WARNING - Movement analysis exceeded expected frames, force completing")
            completeMovementAnalysis()
            return
        }
        
        // Calculate total progress across all enabled phases
        let totalFrames = config.totalCalibrationFrames
        let currentTotalFrames = calibrationFrameCount + movementAnalysisFrameCount
        
        // Report unified progress
        onCalibrationProgress?(currentTotalFrames, totalFrames)
        
        // Collect movement data from current tracks
        for track in trackedObjects {
            let trackId = track.trackId
            let position = track.position
            let confidence = track.score
            
            if var movementData = fishMovementData[trackId] {
                // Update existing track data
                movementData.addPosition(position, confidence: confidence)
                fishMovementData[trackId] = movementData
            } else {
                // Create new track data
                fishMovementData[trackId] = FishMovementData(
                    trackId: trackId,
                    initialPosition: position,
                    confidence: confidence
                )
            }
        }
        
        // Check if we've collected enough movement data
        if movementAnalysisFrameCount >= config.movementAnalysisFrames {
            completeMovementAnalysis()
        }
    }
    
    /// Complete the movement analysis phase (Phase 2)
    @MainActor
    private func completeMovementAnalysis() {
        // Prevent multiple simultaneous completions
        guard isMovementAnalysisPhase && calibrationPhase == .movementAnalysis else {
            return
        }
        
        print("AutoCalibration: Phase 2 complete - Analyzing movement patterns...")
        
        // Immediately mark as no longer in movement analysis to prevent re-entry
        isMovementAnalysisPhase = false
        calibrationPhase = .completed
        
        // Analyze collected movement data
        let movementArray = Array(fishMovementData.values)
        let analysis = MovementAnalyzer.determineDirection(from: movementArray)
        
        print("AutoCalibration: Direction analysis - Confidence: \(analysis.confidence), Qualified tracks: \(analysis.qualifiedTracksCount)")
        
        // Apply detected direction if confidence is sufficient
        if let detectedDirection = analysis.predominantDirection {
            print("AutoCalibration: Auto-detected direction: \(detectedDirection)")
            setCountingDirection(detectedDirection)
            
            // Execute callback safely on main thread
            DispatchQueue.main.async { [weak self] in
                self?.onDirectionDetected?(detectedDirection)
            }
        } else {
            print("AutoCalibration: No clear direction detected, keeping original: \(originalCountingDirection)")
        }
        
        // Complete entire calibration process
        completeEntireCalibration()
        
        // Generate and send calibration summary safely
        let summary = generateCalibrationSummary(analysis: analysis)
        DispatchQueue.main.async { [weak self] in
            self?.onCalibrationSummary?(summary)
        }
    }
    
    /// Complete the entire calibration process
    @MainActor
    private func completeEntireCalibration() {
        // Prevent multiple completions
        guard !isCalibrated else {
            return
        }
        
        // Mark calibration as completed first to prevent re-entry
        isCalibrated = true
        isAutoCalibrationEnabled = false
        isMovementAnalysisPhase = false
        calibrationPhase = .completed
        
        // Clear any previous counting state to ensure new settings are used
        countedTracks.removeAll(keepingCapacity: true)
        crossingDirections.removeAll(keepingCapacity: true)
        previousPositions.removeAll(keepingCapacity: true)
        historyPositions.removeAll(keepingCapacity: true)
        
        // Clear movement analysis data to free memory
        fishMovementData.removeAll(keepingCapacity: true)
        
        // Reset current pixel buffer reference to avoid processing stale data
        currentPixelBuffer = nil
        
        print("AutoCalibration: Complete calibration finished")
    }
    
    /// Generate a comprehensive calibration summary
    @MainActor
    private func generateCalibrationSummary(analysis: DirectionalAnalysis) -> CalibrationSummary {
        let config = AutoCalibrationConfig.shared
        let warnings = MovementAnalyzer.generateWarnings(from: analysis)
        
        // For summary, we need to show display coordinates (same as sliders)
        let displayThresholds: [CGFloat]
        
        if wasPhase1Executed {
            // Phase 1 was executed - thresholds are in counting coordinates from OpenCV
            // Convert them to display coordinates for the summary
            displayThresholds = UnifiedCoordinateSystem.countingToDisplay(
                thresholds, 
                countingDirection: countingDirection
            )
        } else {
            // Phase 1 was bypassed - use the original display thresholds that were stored
            displayThresholds = originalDisplayThresholds.isEmpty ? 
                UnifiedCoordinateSystem.countingToDisplay(thresholds, countingDirection: countingDirection) :
                originalDisplayThresholds
        }
        
        return CalibrationSummary(
            thresholds: displayThresholds,
            detectedDirection: analysis.predominantDirection,
            originalDirection: originalCountingDirection,
            movementAnalysisSuccess: analysis.predominantDirection != nil,
            qualifiedTracksCount: analysis.qualifiedTracksCount,
            warnings: warnings,
            thresholdCalibrationEnabled: config.isThresholdCalibrationEnabled,
            directionCalibrationEnabled: config.isDirectionCalibrationEnabled
        )
    }
    
    /**
     * processObservations
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: Similar to logic in `on_predict_postprocess_end` callback
     * - Takes detection results and passes them to ByteTracker
     * - Updates tracking state and performs counting logic
     */
    override func processObservations(for request: VNRequest, error: Error?) {
        // Save the current pixel buffer for potential calibration use
        let capturedPixelBuffer = currentPixelBuffer
        
        // Process results if available
        if let results = request.results as? [VNRecognizedObjectObservation] {
            // Skip processing entirely if we're in threshold calibration mode
            if isAutoCalibrationEnabled && calibrationPhase == .thresholdDetection {
                return
            }
            
            var boxes: [Box] = []
            var detectionBoxes: [Box] = []
            var scores: [Float] = []
            var labels: [String] = []
            
            // Process detections
            for i in 0..<100 {
                if i < results.count && i < self.numItemsThreshold {
                    let prediction = results[i]
                    let invertedBox = CGRect(
                        x: prediction.boundingBox.minX,
                        y: 1 - prediction.boundingBox.maxY,
                        width: prediction.boundingBox.width,
                        height: prediction.boundingBox.height)
                    let imageRect = VNImageRectForNormalizedRect(
                        invertedBox, Int(inputSize.width), Int(inputSize.height))
                    
                    // Extract detection information
                    let label = prediction.labels[0].identifier
                    let index = self.labels.firstIndex(of: label) ?? 0
                    let confidence = prediction.labels[0].confidence
                    let box = Box(
                        index: index, cls: label, conf: confidence, xywh: imageRect, xywhn: invertedBox)
                    
                    boxes.append(box)
                    detectionBoxes.append(box)
                    scores.append(confidence)
                    labels.append(label)
                }
            }
            
            // Update tracks with new detections using enhanced ByteTracker
            // Use async to prevent blocking, but ensure proper serialization
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                
                // Thread-safe update of tracked objects
                self.trackedObjects = self.byteTracker.update(detections: detectionBoxes, scores: scores, classes: labels)
                
                // Handle movement analysis if we're in Phase 2
                if self.isMovementAnalysisPhase && self.calibrationPhase == .movementAnalysis {
                    self.processMovementAnalysis()
                } else if !self.isAutoCalibrationEnabled && !self.isMovementAnalysisPhase {
                    // Normal counting - check for threshold crossings
                    self.updateCounting()
                }
            }
            
            // Measure FPS
            if self.t1 < 10.0 {  // valid dt
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
            self.t3 = CACurrentMediaTime()
            
            self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)
            
            // Add tracking information to the result
            let result = YOLOResult(
                orig_shape: inputSize, 
                boxes: boxes, 
                speed: self.t2, 
                fps: 1 / self.t4, 
                names: self.labels)
            
            self.currentOnResultsListener?.on(result: result)
        }
    }
    
    /// Process a frame for either calibration or normal inference
    @MainActor
    func processFrame(_ pixelBuffer: CVPixelBuffer) {
        // Store the pixel buffer for use in processObservations
        currentPixelBuffer = pixelBuffer
        
        // If in calibration mode, route to appropriate phase processing
        if isAutoCalibrationEnabled && !isCalibrated {
            switch calibrationPhase {
            case .thresholdDetection:
                processFrameForCalibration(pixelBuffer)
                return  // Only return for threshold detection - skip normal YOLO
            case .movementAnalysis:
                // Movement analysis uses normal YOLO inference, so we let it continue
                // The movement analysis happens in processObservations
                // Don't return here - continue to normal YOLO processing
                break
            case .completed:
                // Should not reach here, but handle gracefully
                break
            }
        }
        
        // For normal processing, the existing Vision pipeline will be used,
        // which will call processObservations
    }
    
    /**
     * updateCounting
     *
     * Maps to Python Implementation:
     * - Primary Correspondence: `check_threshold_crossing` and `check_reverse_threshold_crossing` in counting_demo2.py
     * - Checks if objects have crossed threshold lines
     * - Updates count when objects cross in the designated direction
     * - Prevents double-counting through track state management
     */
    @MainActor
    private func updateCounting() {
        // Skip if in any calibration mode or movement analysis phase
        if isAutoCalibrationEnabled || isMovementAnalysisPhase {
            return
        }
        
        // Get the expected direction of movement for the current counting direction
        let expectedDirection = expectedMovementDirection(for: countingDirection)
        
        // Process all tracks each frame
        for track in trackedObjects {
            let trackId = track.trackId
            let currentPos = track.position
            
            // Skip if no previous position
            guard let lastPosition = previousPositions[trackId] else {
                // If no previous position, just store current and continue
                previousPositions[trackId] = currentPos
                continue
            }
            
            // Store current position for next frame
            previousPositions[trackId] = currentPos
            
            // Calculate movement direction
            let dx = currentPos.x - lastPosition.x
            let dy = currentPos.y - lastPosition.y
            
            // Determine actual movement direction
            let actualDirection: Direction
            if abs(dx) > abs(dy) {
                // Horizontal movement is dominant
                actualDirection = dx > 0 ? .right : .left
            } else {
                // Vertical movement is dominant
                actualDirection = dy > 0 ? .down : .up
            }
            
            // Store the track's movement direction
            crossingDirections[trackId] = actualDirection
            
            // Get previous and current coordinates
            let center_y = currentPos.y
            let last_y = lastPosition.y
            let center_x = currentPos.x
            let last_x = lastPosition.x
            
            // Get counted state for this track
            let alreadyCounted = countedTracks[trackId, default: false]
            
            // Convert positions to unified coordinate system for consistent threshold checking
            let currentUnified = UnifiedCoordinateSystem.UnifiedRect(
                x: center_x, y: center_y, width: 0, height: 0
            )
            let lastUnified = UnifiedCoordinateSystem.UnifiedRect(
                x: last_x, y: last_y, width: 0, height: 0
            )
            
            // Convert to counting coordinates based on direction
            let currentCounting = UnifiedCoordinateSystem.toCounting(currentUnified, countingDirection: countingDirection)
            let lastCounting = UnifiedCoordinateSystem.toCounting(lastUnified, countingDirection: countingDirection)
            
            // Use direction-specific threshold crossing logic
            switch countingDirection {
            case .topToBottom:
                // Top to bottom: fish moving downward (Y increasing)
                let current_y = currentCounting.y
                let last_y = lastCounting.y
                
                // Increment count: Check for crossing from above to below threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_y < threshold && current_y >= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_y > firstThreshold && current_y <= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .bottomToTop:
                // Bottom to top: fish moving upward (Y decreasing)
                let current_y = currentCounting.y
                let last_y = lastCounting.y
                
                // Increment count: Check for crossing from below to above threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_y > threshold && current_y <= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_y < firstThreshold && current_y >= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .leftToRight:
                // Left to right: fish moving rightward (X increasing)
                let current_x = currentCounting.x
                let last_x = lastCounting.x
                
                // Increment count: Check for crossing from left to right of threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_x < threshold && current_x >= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_x > firstThreshold && current_x <= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .rightToLeft:
                // Right to left: fish moving leftward (X decreasing)
                let current_x = currentCounting.x
                let last_x = lastCounting.x
                
                // Increment count: Check for crossing from right to left of threshold
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_x > threshold && current_x <= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_x < firstThreshold && current_x >= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
            }
        }
        
        // Increment frame count for each update
        frameCount += 1
        
        // Clean up old data periodically (every 30 frames)
        if frameCount % 30 == 0 {
            // Create a set once for efficient lookups
            let currentIds = Set(trackedObjects.map { $0.trackId })
            
            // Remove keys for tracks that no longer exist
            let keysToRemove = countedTracks.keys.filter { !currentIds.contains($0) }
            for key in keysToRemove {
                countedTracks.removeValue(forKey: key)
                previousPositions.removeValue(forKey: key)
                historyPositions.removeValue(forKey: key)
                crossingDirections.removeValue(forKey: key)
            }
        }
    }
    
    /**
     * countObject
     *
     * Maps to Python Implementation:
     * - In Python, this is part of the `check_threshold_crossing` function
     * - Increments count and updates tracking state
     * - Provides visualization feedback
     */
    @MainActor
    private func countObject(trackId: Int) {
        // Only count if not already counted
        if countedTracks[trackId] != true {
            totalCount += 1
            countedTracks[trackId] = true
            
            // Mark the track as counted - only search if needed
            if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                trackedObjects[trackIndex].markCounted()
            }
        }
    }
    
    /// Gets the tracking information for a detection box
    ///
    /// - Parameters:
    ///   - box: The detection box to check
    /// - Returns: A tuple containing (isTracked, isCounted)
    @MainActor
    func getTrackingStatus(for box: Box) -> (isTracked: Bool, isCounted: Bool) {
        if let trackInfo = getTrackInfo(for: box) {
            return (true, trackInfo.isCounted)
        }
        return (false, false)
    }
    
    /**
     * getTrackInfo
     *
     * Maps to Python Implementation:
     * - Used for visualization, similar to coloring logic in Python's drawing functions
     * - Identifies track ID and counted status for UI rendering
     */
    @MainActor
    func getTrackInfo(for box: Box) -> (trackId: Int, isCounted: Bool)? {
        // Calculate the center of the box
        let centerX = (box.xywhn.minX + box.xywhn.maxX) / 2
        let centerY = (box.xywhn.minY + box.xywhn.maxY) / 2
        
        // Find tracks that match this box
        var bestMatch: STrack? = nil
        var minDistance: CGFloat = TrackingParameters.minMatchDistance
        
        // First try to match by IOU (Intersection over Union)
        for track in trackedObjects {
            guard let trackBox = track.lastDetection else { continue }
            
            // Calculate IoU between the track's box and the current box
            let boxA = box.xywhn
            let boxB = trackBox.xywhn
            
            // Calculate intersection area
            let xA = max(boxA.minX, boxB.minX)
            let yA = max(boxA.minY, boxB.minY)
            let xB = min(boxA.maxX, boxB.maxX)
            let yB = min(boxA.maxY, boxB.maxY)
            
            let interArea = max(0, xB - xA) * max(0, yB - yA)
            
            // Calculate union area
            let boxAArea = boxA.width * boxA.height
            let boxBArea = boxB.width * boxB.height
            let unionArea = boxAArea + boxBArea - interArea
            
            let iou = unionArea > 0 ? Float(interArea / unionArea) : 0.0
            
            // For high IoU, immediately select this track
            if iou > Float(TrackingParameters.iouMatchThreshold) {
                return (trackId: track.trackId, isCounted: countedTracks[track.trackId] ?? false)
            }
            
            // Otherwise, continue with the distance-based approach
            let dx = track.position.x - centerX
            let dy = track.position.y - centerY
            let distance = sqrt(dx*dx + dy*dy)
            
            if distance < minDistance {
                minDistance = distance
                bestMatch = track
            }
        }
        
        // If we found a good match by distance
        if let track = bestMatch {
            return (trackId: track.trackId, isCounted: countedTracks[track.trackId] ?? false)
        }
        
        return nil
    }
    
    /// Checks if a detection box is currently being tracked
    ///
    /// - Parameter box: The detection box to check
    /// - Returns: True if the box is associated with an active track
    @MainActor
    func isObjectTracked(box: Box) -> Bool {
        return getTrackingStatus(for: box).isTracked
    }
    
    /// Checks if a detection box has been counted
    ///
    /// - Parameter box: The detection box to check
    /// - Returns: True if the box is associated with a track that has been counted
    @MainActor
    func isObjectCounted(box: Box) -> Bool {
        return getTrackingStatus(for: box).isCounted
    }
    
    /// Helper method to get the expected movement direction based on counting direction
    private func expectedMovementDirection(for countingDirection: CountingDirection) -> Direction {
        switch countingDirection {
        case .topToBottom:
            return .down
        case .bottomToTop:
            return .up
        case .leftToRight:
            return .right
        case .rightToLeft:
            return .left
        }
    }
    
    // MARK: - Enhanced Threshold Management
    
    /// Gets the current threshold values
    ///
    /// - Returns: Array of threshold values (usually 2 values)
    @MainActor
    func getThresholds() -> [CGFloat] {
        return thresholds
    }
    
    /// Check if auto-calibration is currently enabled
    ///
    /// - Returns: True if auto-calibration is active
    @MainActor
    func getAutoCalibrationEnabled() -> Bool {
        return isAutoCalibrationEnabled
    }
    
    /// Check if calibration has been completed
    ///
    /// - Returns: True if calibration has finished
    @MainActor
    func getCalibrationStatus() -> Bool {
        return isCalibrated
    }
    
    /// Get the target number of frames for calibration
    ///
    /// - Returns: The total number of frames needed for calibration
    @MainActor
    func getTargetCalibrationFrames() -> Int {
        return targetCalibrationFrames
    }
    
    /// Set the target number of frames for calibration
    ///
    /// - Parameter count: The number of frames to use for calibration
    @MainActor
    func setTargetCalibrationFrames(_ count: Int) {
        targetCalibrationFrames = max(30, count) // Ensure at least 30 frames
    }
}
