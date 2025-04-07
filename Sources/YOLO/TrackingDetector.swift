// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, implementing object tracking functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The TrackingDetector class extends ObjectDetector with tracking capabilities using ByteTrack.
//  It tracks objects across frames, associates detections with existing tracks, and supports
//  threshold crossing detection for counting applications like fish counting. The implementation
//  maintains tracking state for each detected object including position history and counting status.

import Foundation
import UIKit
import Vision

/// Direction for counting fish
public enum CountingDirection {
    case topToBottom
    case bottomToTop
}

/// Direction of fish movement
private enum Direction {
    case up
    case down
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
    
    /// Thresholds used for counting (normalized y-coordinates, 0.0-1.0)
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
        countedTracks.removeAll()
        crossingDirections.removeAll()
        previousPositions.removeAll()
        historyPositions.removeAll()
        frameCount = 0
        
        // Reset the ByteTracker to reset the track IDs
        byteTracker.reset()
        trackedObjects.removeAll()
    }
    
    /// Sets the counting direction
    @MainActor
    func setCountingDirection(_ direction: CountingDirection) {
        self.countingDirection = direction
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
        if let results = request.results as? [VNRecognizedObjectObservation] {
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
        let upperThreshold = thresholds.first ?? 0.3
        let lowerThreshold = thresholds.last ?? 0.5
        
        // Define a larger buffer zone to make detection more sensitive
        let thresholdBufferZone: CGFloat = 0.03 // Increased from 0.02 to 0.04 (4% of screen height)
        
        // Increment frame count for each update
        frameCount += 1
        
        // Use more efficient iteration - consider all tracks for counting now, including newly created ones
        for track in trackedObjects {
            let trackId = track.trackId
            
            // Skip tracks that have already been counted
            guard countedTracks[trackId] != true else { continue }
            
            let y = track.position.y
            
            // Get previous position
            guard let lastPosition = previousPositions[trackId] else {
                // If no previous position, just store current and continue
                previousPositions[trackId] = track.position
                continue
            }
            
            let lastY = lastPosition.y
            
            // Store the 3-frame history position for detecting fast movements (more frequent updates)
            if frameCount % 3 == 0 {
                historyPositions[trackId] = lastPosition
            }
            
            // Get history position for detecting fast movements
            let historyY = historyPositions[trackId]?.y ?? lastY
            
            // Store current position for next frame
            previousPositions[trackId] = track.position
            
            // Handle detection based on counting direction
            switch countingDirection {
            case .topToBottom:
                // Count when crossing or approaching ANY threshold line
                // More sensitive version with wider buffer zone
                if (lastY < upperThreshold && y >= upperThreshold - thresholdBufferZone) || 
                   (lastY < lowerThreshold && y >= lowerThreshold - thresholdBufferZone) {
                    countObject(trackId: trackId)
                }
                // Count if position is close to threshold and moving downward (even more sensitive)
                else if ((abs(y - upperThreshold) < thresholdBufferZone * 1.5 && lastY < y) ||
                        (abs(y - lowerThreshold) < thresholdBufferZone * 1.5 && lastY < y)) &&
                        track.trackletLen > 3 { // Reduced from 5 to 3 for quicker counting
                    countObject(trackId: trackId)
                }
                // Detect rapid movements that might have skipped the threshold check (more sensitive)
                else if historyY < upperThreshold && y > upperThreshold && 
                        track.trackletLen > 7 { // Reduced from 10 to 5
                    countObject(trackId: trackId)
                }
                else if historyY < lowerThreshold && y > lowerThreshold && 
                        track.trackletLen > 7 { // Reduced from 10 to 5
                    countObject(trackId: trackId)
                }
                // Count if fish is currently between thresholds and moving downward
                else if y > upperThreshold && y < lowerThreshold && lastY < y && 
                        track.trackletLen > 8 {
                    countObject(trackId: trackId)
                }
                // Count if fish is currently below lower threshold and was recently activated
                else if y > lowerThreshold && track.isActivated && track.trackletLen > 5 && track.trackletLen < 15 {
                    countObject(trackId: trackId)
                }
                // Handle reverse crossing only for first threshold
                else if lastY > upperThreshold + thresholdBufferZone && y <= upperThreshold - thresholdBufferZone && countedTracks[trackId] == true {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                }
                
            case .bottomToTop:
                // Count when crossing or approaching ANY threshold line
                // More sensitive version with wider buffer zone
                if (lastY > upperThreshold && y <= upperThreshold + thresholdBufferZone) || 
                   (lastY > lowerThreshold && y <= lowerThreshold + thresholdBufferZone) {
                    countObject(trackId: trackId)
                }
                // Count if position is close to threshold and moving upward (even more sensitive)
                else if ((abs(y - upperThreshold) < thresholdBufferZone * 1.5 && lastY > y) ||
                        (abs(y - lowerThreshold) < thresholdBufferZone * 1.5 && lastY > y)) &&
                        track.trackletLen > 3 { // Reduced from 5 to 3 for quicker counting
                    countObject(trackId: trackId)
                }
                // Detect rapid movements that might have skipped the threshold check (more sensitive)
                else if historyY > upperThreshold && y < upperThreshold && 
                        track.trackletLen > 5 { // Reduced from 10 to 5
                    countObject(trackId: trackId)
                }
                else if historyY > lowerThreshold && y < lowerThreshold && 
                        track.trackletLen > 5 { // Reduced from 10 to 5
                    countObject(trackId: trackId)
                }
                // Count if fish is currently between thresholds and moving upward
                else if y < lowerThreshold && y > upperThreshold && lastY > y && 
                        track.trackletLen > 8 {
                    countObject(trackId: trackId)
                }
                // Count if fish is currently above upper threshold and was recently activated
                else if y < upperThreshold && track.isActivated && track.trackletLen > 5 && track.trackletLen < 15 {
                    countObject(trackId: trackId)
                }
                // Handle reverse crossing only for first threshold
                else if lastY < lowerThreshold - thresholdBufferZone && y >= lowerThreshold + thresholdBufferZone && countedTracks[trackId] == true {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                }
            }
        }
        
        // Count any stable tracks that have been around for a while and are in the "counted zone"
        // This catches fish that somehow never triggered the crossing detection
        // More frequent checking (every second) with shorter required tracking time
        if frameCount % 30 == 0 { // Every ~1 second at 30fps (changed from 60 to 30)
            for track in trackedObjects where track.state == .tracked && track.trackletLen > 15 && !countedTracks[track.trackId, default: false] {
                let y = track.position.y
                
                switch countingDirection {
                case .topToBottom:
                    // If track is beyond the lowermost threshold - reduced required distance
                    if y > lowerThreshold + 0.05 { // 5% beyond the threshold (reduced from 10%)
                        countObject(trackId: track.trackId)
                    }
                    // Or if it's between thresholds
                    else if y > upperThreshold + 0.05 && y < lowerThreshold {
                        countObject(trackId: track.trackId)
                    }
                case .bottomToTop:
                    // If track is above the uppermost threshold - reduced required distance
                    if y < upperThreshold - 0.05 { // 5% above the threshold (reduced from 10%)
                        countObject(trackId: track.trackId)
                    }
                    // Or if it's between thresholds
                    else if y < lowerThreshold - 0.05 && y > upperThreshold {
                        countObject(trackId: track.trackId)
                    }
                }
            }
        }
        
        // Perform a comprehensive check for any tracks that should be counted but weren't
        if frameCount % 120 == 0 { // Every ~4 seconds at 30fps
            // Get highest track ID to estimate how many tracks we've seen
            let maxTrackId = trackedObjects.map { $0.trackId }.max() ?? 0
            
            // If we have significantly more track IDs than counted fish, be more aggressive with counting
            if (maxTrackId > Int(Double(totalCount) * 1.2)) && maxTrackId > 10 {
                for track in trackedObjects where track.state == .tracked && !countedTracks[track.trackId, default: false] {
                    // Count any tracked fish that's been visible for a while
                    if track.trackletLen > 10 {
                        countObject(trackId: track.trackId)
                    }
                }
            }
        }
        
        // Clean up tracking for objects that are no longer visible - more efficient approach
        if frameCount % 30 == 0 {  // Only clean up every 30 frames to reduce overhead
            let currentIds = Set(trackedObjects.map { $0.trackId })
            
            for key in countedTracks.keys where !currentIds.contains(key) {
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
            
            // Mark the track as counted
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
        var minDistance: CGFloat = 0.3 // Increased from 0.2 to be more lenient with camera movement
        
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
            
            let iou = unionArea > 0 ? interArea / unionArea : 0
            
            // For high IoU, immediately select this track
            if iou > 0.4 { // Lowered from 0.5 to be more lenient
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
}
