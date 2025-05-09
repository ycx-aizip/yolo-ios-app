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
    @MainActor private let byteTracker = ByteTracker()
    
    /// Total count of objects that have crossed the threshold(s)
    private var totalCount: Int = 0
    
    /// Thresholds used for counting (normalized coordinates, 0.0-1.0)
    /// For vertical directions (topToBottom, bottomToTop), these are y-coordinates
    /// For horizontal directions (leftToRight, rightToLeft), these are x-coordinates
    private var thresholds: [CGFloat] = [0.3, 0.5]
    
    /// Map of track IDs to counting status
    private var countedTracks: [Int: Bool] = [:]
    
    /// The current tracked objects
    private var trackedObjects: [STrack] = []
    
    /// Current counting direction
    private var countingDirection: CountingDirection = .topToBottom
    
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
    
    /// Buffer to store frames during calibration
    private var calibrationFrameBuffer: [CVPixelBuffer] = []
    
    /// Number of frames to collect for calibration (approx. 5 seconds at 30fps)
    private var calibrationFrameCount: Int = CalibrationUtils.defaultCalibrationFrameCount
    
    /// Reference to the current frame being processed
    private var currentPixelBuffer: CVPixelBuffer?
    
    /// Callback for reporting calibration progress
    var onCalibrationProgress: ((Int, Int) -> Void)?
    
    /// Callback for reporting calibration completion with new thresholds
    var onCalibrationComplete: (([CGFloat]) -> Void)?
    
    // MARK: - Threshold management
    
    /// Sets the thresholds for counting
    @MainActor
    func setThresholds(_ values: [CGFloat]) {
        guard values.count >= 1 else { return }
        
        // Ensure thresholds are within valid range (0.0-1.0)
        let validThresholds = values.map { max(0.0, min(1.0, $0)) }
        self.thresholds = validThresholds
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
        // If we're disabling calibration that was previously enabled, make sure we clean up properly
        if !enabled && isAutoCalibrationEnabled {
            // Clear out any partial calibration data
            calibrationFrameBuffer.removeAll(keepingCapacity: true)
            currentPixelBuffer = nil
            isCalibrated = false
        }
        
        // Set the new state
        isAutoCalibrationEnabled = enabled
        
        if enabled {
            // Reset calibration state when enabling
            resetCalibration()
        }
    }
    
    /// Reset calibration state
    @MainActor
    private func resetCalibration() {
        isCalibrated = false
        calibrationFrameBuffer.removeAll(keepingCapacity: true)
    }
    
    /// Add a frame to the calibration buffer
    @MainActor
    private func addFrameToCalibrationBuffer(_ pixelBuffer: CVPixelBuffer) {
        // Skip if we already have enough frames or calibration is completed
        if calibrationFrameBuffer.count >= calibrationFrameCount || isCalibrated {
            return
        }
        
        // Add the frame to the buffer
        calibrationFrameBuffer.append(pixelBuffer)
        
        // Report progress via callback
        let progress = calibrationFrameBuffer.count
        onCalibrationProgress?(progress, calibrationFrameCount)
        
        // Perform calibration if we have collected enough frames
        if calibrationFrameBuffer.count >= calibrationFrameCount {
            performCalibration()
        }
    }
    
    /// Perform calibration calculation on collected frames
    @MainActor
    private func performCalibration() {
        // Calculate new thresholds from collected frames
        let (threshold1, threshold2) = CalibrationUtils.calculateThresholds(
            from: calibrationFrameBuffer,
            direction: countingDirection
        )
        
        // Set the new thresholds
        thresholds = [threshold1, threshold2]
        
        // Mark calibration as completed
        isCalibrated = true
        isAutoCalibrationEnabled = false
        
        // Clear buffer to free memory
        calibrationFrameBuffer.removeAll(keepingCapacity: true)
        
        // Clear any previous counting state to ensure new thresholds are used
        countedTracks.removeAll(keepingCapacity: true)
        crossingDirections.removeAll(keepingCapacity: true)
        previousPositions.removeAll(keepingCapacity: true)
        historyPositions.removeAll(keepingCapacity: true)
        
        // Reset current pixel buffer reference to avoid processing stale data
        currentPixelBuffer = nil
        
        // Notify completion via callback
        onCalibrationComplete?(thresholds)
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
            // First check if we're in calibration mode
            if isAutoCalibrationEnabled, let pixelBuffer = capturedPixelBuffer {
                // Add the frame to calibration buffer on the main actor
                Task { @MainActor in
                    self.addFrameToCalibrationBuffer(pixelBuffer)
                }
                
                // Skip normal detection processing during calibration
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
            Task { @MainActor in
                trackedObjects = byteTracker.update(detections: detectionBoxes, scores: scores, classes: labels)
                // Check for threshold crossings
                updateCounting()
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
        
        // If in calibration mode, process directly here
        if isAutoCalibrationEnabled {
            addFrameToCalibrationBuffer(pixelBuffer)
            return
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
        // Skip if in calibration mode
        if isAutoCalibrationEnabled {
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
            
            // Only check appropriate thresholds based on the current counting direction
            switch countingDirection {
            case .topToBottom:
                // Increment count: Check for threshold crossing from top to bottom
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_y < threshold && center_y >= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing from bottom to top (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_y > firstThreshold && center_y <= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .bottomToTop:
                // Increment count: Check for threshold crossing from bottom to top
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_y > threshold && center_y <= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_y < firstThreshold && center_y >= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .leftToRight:
                // Increment count: Check for threshold crossing from left to right
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_x < threshold && center_x >= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_x > firstThreshold && center_x <= firstThreshold && alreadyCounted {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .rightToLeft:
                // Increment count: Check for threshold crossing from right to left
                if !alreadyCounted {
                    for threshold in thresholds {
                        if last_x > threshold && center_x <= threshold {
                            countObject(trackId: trackId)
                            break // Count only once per threshold crossing event
                        }
                    }
                }
                
                // Decrement count: Check for reverse crossing (only first threshold)
                if let firstThreshold = thresholds.first,
                   last_x < firstThreshold && center_x >= firstThreshold && alreadyCounted {
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
}
