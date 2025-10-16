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
/// NOTE: Counting-related config (thresholds, direction) is now in ThresholdCounter
@MainActor
public class TrackingDetectorConfig {
    /// Shared singleton instance
    public static let shared = TrackingDetectorConfig()
    
    /// Default confidence threshold for YOLO detection
    public var defaultConfidenceThreshold: Float = 0.60
    
    /// Default IoU threshold for non-maximum suppression
    public var defaultIoUThreshold: Float = 0.50
    
    /// Default number of items threshold for detection display
    public var defaultNumItemsThreshold: Int = 100
    
    private init() {
        // Private initializer to enforce singleton pattern
    }
    
    /// Update detection-related default values
    /// NOTE: For counting config, use ThresholdCounter.defaultThresholds and ThresholdCounter.defaultCountingDirection
    public func updateDefaults(
        confidenceThreshold: Float? = nil,
        iouThreshold: Float? = nil,
        numItemsThreshold: Int? = nil
    ) {
        if let confidenceThreshold = confidenceThreshold {
            self.defaultConfidenceThreshold = confidenceThreshold
        }
        if let iouThreshold = iouThreshold {
            self.defaultIoUThreshold = iouThreshold
        }
        if let numItemsThreshold = numItemsThreshold {
            self.defaultNumItemsThreshold = numItemsThreshold
        }

        print("TrackingDetector: Detection configuration updated")
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
    
    /// The tracker instance (protocol-based for flexibility)
    /// Using nonisolated(unsafe) to avoid Sendable issues with protocol types
    nonisolated(unsafe) private var tracker: TrackerProtocol
    
    /// The counter instance (protocol-based for flexibility)
    /// Using nonisolated(unsafe) to avoid Sendable issues with protocol types
    nonisolated(unsafe) private var counter: CounterProtocol
    
    /// The current tracked objects
    private var trackedObjects: [STrack] = []
    
    /// Current counting direction (maintained for calibration)
    private var countingDirection: CountingDirection
    
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
    
    /// Temporary storage for calibration thresholds
    private var calibrationThresholds: [CGFloat] = []
    
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
    
    /// Initialize TrackingDetector with optional dependency injection
    /// Configuration will be applied after initialization via applySharedConfiguration()
    /// - Parameters:
    ///   - tracker: Optional tracker implementation (defaults to OCSort with coreMLOCSortConfig)
    ///   - counter: Optional counter implementation (defaults to ThresholdCounter)
    required init(tracker: TrackerProtocol? = nil, counter: CounterProtocol? = nil) {
        // Initialize with dependency injection
        // Default: OCSort with CoreML-tuned configuration
        // self.tracker = tracker ?? OCSort()
        self.tracker = tracker ?? ByteTracker()
        self.counter = counter ?? ThresholdCounter()
        self.countingDirection = .bottomToTop
        self.originalCountingDirection = .bottomToTop

        super.init()
    }
    
    /// Required override for ObjectDetector compatibility
    required convenience init() {
        self.init(tracker: nil, counter: nil)
    }
    
    /// Apply the shared configuration to this instance
    /// This must be called after initialization to apply the centralized settings
    @MainActor
    func applySharedConfiguration() {
        let config = TrackingDetectorConfig.shared

        // Apply detection configuration
        self.confidenceThreshold = Double(config.defaultConfidenceThreshold)
        self.iouThreshold = Double(config.defaultIoUThreshold)
        self.numItemsThreshold = config.defaultNumItemsThreshold

        // Apply counting configuration from ThresholdCounter
        self.countingDirection = ThresholdCounter.defaultCountingDirection

        // Configure the counter with counting defaults
        counter.configure(thresholds: ThresholdCounter.defaultThresholds, direction: ThresholdCounter.defaultCountingDirection)
        
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
        
        // Delegate to counter
        counter.configure(thresholds: values, direction: countingDirection)
    }
    
    /// Sets the thresholds with original display values (for bypass mode)
    @MainActor
    func setThresholds(_ countingValues: [CGFloat], originalDisplayValues: [CGFloat]) {
        guard countingValues.count >= 1 && originalDisplayValues.count >= 1 else { return }
        
        // Configure counter with counting thresholds
        counter.configure(thresholds: countingValues, direction: countingDirection)
        
        // Store original display thresholds for bypass mode (for calibration)
        let validDisplayThresholds = originalDisplayValues.map { max(0.0, min(1.0, $0)) }
        self.originalDisplayThresholds = validDisplayThresholds
    }
    
    /// Gets the current count of objects that have crossed the threshold
    ///
    /// - Returns: The total count
    @MainActor
    func getCount() -> Int {
        return counter.getTotalCount()
    }
    
    /// Resets the counting state and clears all tracked objects
    @MainActor
    public func resetCount() {
        // Delegate to counter
        counter.resetCount()
        
        // Clean up any tracked objects to release memory
        for track in trackedObjects {
            track.cleanup()
        }
        
        // Reset the tracker to reset the track IDs
        tracker.reset()
        trackedObjects.removeAll(keepingCapacity: true)
    }
    
    /// Sets the counting direction
    @MainActor
    func setCountingDirection(_ direction: CountingDirection) {
        self.countingDirection = direction
        
        // Update counter configuration
        counter.configure(thresholds: getThresholds(), direction: direction)
        
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
            
            // Reset counter state
            counter.resetCount()
            
            // Reset the tracker to clear any existing tracks
            tracker.reset()
            
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
                
                print("AutoCalibration: Phase 1 bypassed - Using current thresholds: \(getThresholds())")
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
                calibrationThresholds = [threshold1, threshold2]
            } else {
                // Accumulate by blending with previous values using weighted average
                // Give more weight to newer frames using a simple exponential moving average
                let weight = 2.0 / Double(calibrationFrameCount + 1) 
                let newThreshold1 = CGFloat(weight) * threshold1 + CGFloat(1 - weight) * calibrationThresholds[0]
                let newThreshold2 = CGFloat(weight) * threshold2 + CGFloat(1 - weight) * calibrationThresholds[1]
                calibrationThresholds = [newThreshold1, newThreshold2]
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
        
        // Determine final thresholds
        let finalThresholds: [CGFloat]
        if wasPhase1Executed {
            // Sort thresholds to ensure proper order for OpenCV-detected values
            finalThresholds = calibrationThresholds.sorted()
            print("AutoCalibration: Phase 1 complete (executed) - Thresholds sorted: \(finalThresholds)")
        } else {
            // Phase 1 was bypassed - keep current thresholds exactly as set by user
            finalThresholds = getThresholds()
            print("AutoCalibration: Phase 1 complete (bypassed) - Thresholds preserved: \(finalThresholds)")
        }
        
        // Update counter with final thresholds
        counter.configure(thresholds: finalThresholds, direction: countingDirection)
        
        // Notify threshold completion
        onCalibrationComplete?(finalThresholds)
        
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
        
        // Reset counter state
        counter.resetCount()
        
        // Step 3: Reset tracker synchronously
        tracker.reset()
        
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
        
        // Reset counter to ensure new settings are used
        counter.resetCount()
        
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
                getThresholds(), 
                countingDirection: countingDirection
            )
        } else {
            // Phase 1 was bypassed - use the original display thresholds that were stored
            displayThresholds = originalDisplayThresholds.isEmpty ? 
                UnifiedCoordinateSystem.countingToDisplay(getThresholds(), countingDirection: countingDirection) :
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
            
            // Update tracks with new detections using tracker
            // Use async to prevent blocking, but ensure proper serialization
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                
                // Thread-safe update of tracked objects
                self.trackedObjects = self.tracker.update(detections: detectionBoxes, scores: scores, classes: labels)
                
                // Handle movement analysis if we're in Phase 2
                if self.isMovementAnalysisPhase && self.calibrationPhase == .movementAnalysis {
                    self.processMovementAnalysis()
                } else if !self.isAutoCalibrationEnabled && !self.isMovementAnalysisPhase {
                    // Normal counting - process frame through counter
                    let countingResult = self.counter.processFrame(tracks: self.trackedObjects)
                    // Counter handles all counting logic internally
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
    
    // NOTE: updateCounting() and countObject() methods have been removed.
    // Counting logic is now handled by the CounterProtocol implementation (ThresholdCounter).
    // This change enables pluggable counting strategies and cleaner separation of concerns.
    
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
        
        // Get tracking info from counter
        let trackingInfo = counter.getTrackingInfo()
        
        // Find tracks that match this box
        var bestMatch: (trackId: Int, isCounted: Bool)? = nil
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
                // Find counted status from counter's tracking info
                let isCounted = trackingInfo.first(where: { $0.trackId == track.trackId })?.isCounted ?? false
                return (trackId: track.trackId, isCounted: isCounted)
            }
            
            // Otherwise, continue with the distance-based approach
            let dx = track.position.x - centerX
            let dy = track.position.y - centerY
            let distance = sqrt(dx*dx + dy*dy)
            
            if distance < minDistance {
                minDistance = distance
                let isCounted = trackingInfo.first(where: { $0.trackId == track.trackId })?.isCounted ?? false
                bestMatch = (trackId: track.trackId, isCounted: isCounted)
            }
        }
        
        return bestMatch
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
    
    // MARK: - Enhanced Threshold Management
    
    /// Gets the current threshold values
    ///
    /// - Returns: Array of threshold values (usually 2 values)
    @MainActor
    func getThresholds() -> [CGFloat] {
        // Get thresholds from counter's tracking info
        // Since counter doesn't expose thresholds directly, we use calibrationThresholds if available
        // Otherwise return default from ThresholdCounter
        if !calibrationThresholds.isEmpty {
            return calibrationThresholds
        }
        return ThresholdCounter.defaultThresholds
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

