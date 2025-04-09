// from Aizip

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
        
        // Process only a subset of tracks when dealing with many fish to reduce CPU load
        // This is based on the observation that not all tracks need to be evaluated every frame
        let maxTracksToProcess = 30 // Cap the number of tracks we evaluate in a single frame
        
        // If we have too many tracks, prioritize processing active ones that haven't been counted yet
        var tracksToProcess = trackedObjects
        if trackedObjects.count > maxTracksToProcess {
            // First prioritize tracks that are near thresholds
            tracksToProcess = trackedObjects.filter { track in
                let y = track.position.y
                return !countedTracks[track.trackId, default: false] && 
                       (abs(y - upperThreshold) < 0.15 || abs(y - lowerThreshold) < 0.15)
            }
            
            // If we still have too many, take the most recently updated tracks
            if tracksToProcess.count > maxTracksToProcess {
                tracksToProcess.sort { $0.endFrame > $1.endFrame }
                tracksToProcess = Array(tracksToProcess.prefix(maxTracksToProcess))
            }
            
            // If we have very few tracks to process, add some already counted tracks for reverse counting
            if tracksToProcess.count < 10 {
                let countedTracksList = trackedObjects.filter { countedTracks[$0.trackId, default: false] }
                    .prefix(maxTracksToProcess - tracksToProcess.count)
                tracksToProcess.append(contentsOf: countedTracksList)
            }
        }
        
        // Process the selected tracks
        for track in tracksToProcess {
            let trackId = track.trackId
            let y = track.position.y
            
            // Get previous position
            guard let lastPosition = previousPositions[trackId] else {
                // If no previous position, just store current and continue
                previousPositions[trackId] = track.position
                continue
            }
            
            let lastY = lastPosition.y
            
            // Only store history positions every N frames to reduce memory updates
            if trackId % 3 == frameCount % 3 {  // Distribute updates across frames
                historyPositions[trackId] = lastPosition
            }
            
            // Get history position for detecting fast movements
            let historyY = historyPositions[trackId]?.y ?? lastY
            
            // Store current position for next frame
            previousPositions[trackId] = track.position
            
            // Skip counting logic if already counted (except for reverse counting)
            let alreadyCounted = countedTracks[trackId, default: false]
            
            // Handle detection based on counting direction
            switch countingDirection {
            case .topToBottom:
                // Simplified conditional checks - consolidate similar conditions
                if !alreadyCounted {
                    // Primary conditions for counting (from top to bottom)
                    let crossedUpper = lastY < upperThreshold && y >= upperThreshold - thresholdBufferZone
                    let crossedLower = lastY < lowerThreshold && y >= lowerThreshold - thresholdBufferZone
                    let nearUpperMovingDown = abs(y - upperThreshold) < thresholdBufferZone * 1.5 && lastY < y && track.trackletLen > 3
                    let nearLowerMovingDown = abs(y - lowerThreshold) < thresholdBufferZone * 1.5 && lastY < y && track.trackletLen > 3
                    let rapidlyPassedUpper = historyY < upperThreshold && y > upperThreshold && track.trackletLen > 7
                    let rapidlyPassedLower = historyY < lowerThreshold && y > lowerThreshold && track.trackletLen > 7
                    let betweenThresholdsMovingDown = y > upperThreshold && y < lowerThreshold && lastY < y && track.trackletLen > 5
                    let belowLowerThreshold = y > lowerThreshold && track.isActivated && track.trackletLen > 5 && track.trackletLen < 15
                    
                    // Combine all counting conditions with a single if statement
                    if crossedUpper || crossedLower || 
                       nearUpperMovingDown || nearLowerMovingDown ||
                       rapidlyPassedUpper || rapidlyPassedLower ||
                       betweenThresholdsMovingDown || belowLowerThreshold {
                        countObject(trackId: trackId)
                    }
                }
                // Handle reverse crossing (uncounting) - only check this for counted fish
                else if lastY > upperThreshold + thresholdBufferZone && y <= upperThreshold - thresholdBufferZone {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
                
            case .bottomToTop:
                // Simplified conditional checks - consolidate similar conditions
                if !alreadyCounted {
                    // Primary conditions for counting (from bottom to top)
                    let crossedUpper = lastY > upperThreshold && y <= upperThreshold + thresholdBufferZone
                    let crossedLower = lastY > lowerThreshold && y <= lowerThreshold + thresholdBufferZone
                    let nearUpperMovingUp = abs(y - upperThreshold) < thresholdBufferZone * 1.5 && lastY > y && track.trackletLen > 3
                    let nearLowerMovingUp = abs(y - lowerThreshold) < thresholdBufferZone * 1.5 && lastY > y && track.trackletLen > 3
                    let rapidlyPassedUpper = historyY > upperThreshold && y < upperThreshold && track.trackletLen > 5
                    let rapidlyPassedLower = historyY > lowerThreshold && y < lowerThreshold && track.trackletLen > 5
                    let betweenThresholdsMovingUp = y < lowerThreshold && y > upperThreshold && lastY > y && track.trackletLen > 5
                    let aboveUpperThreshold = y < upperThreshold && track.isActivated && track.trackletLen > 5 && track.trackletLen < 15
                    
                    // Combine all counting conditions with a single if statement
                    if crossedUpper || crossedLower || 
                       nearUpperMovingUp || nearLowerMovingUp ||
                       rapidlyPassedUpper || rapidlyPassedLower ||
                       betweenThresholdsMovingUp || aboveUpperThreshold {
                        countObject(trackId: trackId)
                    }
                }
                // Handle reverse crossing (uncounting) - only check this for counted fish
                else if lastY < lowerThreshold - thresholdBufferZone && y >= lowerThreshold + thresholdBufferZone {
                    totalCount = max(0, totalCount - 1)
                    countedTracks[trackId] = false
                    if let trackIndex = trackedObjects.firstIndex(where: { $0.trackId == trackId }) {
                        trackedObjects[trackIndex].counted = false
                    }
                }
            }
        }
        
        // Count stable tracks less frequently to reduce CPU load
        // This catches fish that somehow never triggered the crossing detection
        if frameCount % 60 == 0 { // Reduced frequency from 30 to 60 frames
            // Limit the number of tracks to check
            let stableTracks = trackedObjects.filter { 
                $0.state == .tracked && 
                $0.trackletLen > 15 && 
                !countedTracks[$0.trackId, default: false] 
            }.prefix(20) // Only check up to 20 tracks
            
            for track in stableTracks {
                let y = track.position.y
                
                switch countingDirection {
                case .topToBottom:
                    // Simplified conditions for stable tracks
                    if y > lowerThreshold + 0.05 || (y > upperThreshold + 0.05 && y < lowerThreshold) {
                        countObject(trackId: track.trackId)
                    }
                case .bottomToTop:
                    // Simplified conditions for stable tracks
                    if y < upperThreshold - 0.05 || (y < lowerThreshold - 0.05 && y > upperThreshold) {
                        countObject(trackId: track.trackId)
                    }
                }
            }
        }
        
        // Perform comprehensive check much less frequently
        // Only run this when we might have missed some fish
        if frameCount % 180 == 0 { // Reduced from 120 to 180 frames
            // Use quick approximation instead of expensive max operation
            let maxTrackId = trackedObjects.isEmpty ? 0 : trackedObjects.last?.trackId ?? 0
            
            // If we have significantly more track IDs than counted fish, be more aggressive with counting
            if (maxTrackId > Int(Double(totalCount) * 1.2)) && maxTrackId > 10 {
                // Limit the number of tracks to check
                let tracksToCheck = trackedObjects.filter { 
                    $0.state == .tracked && 
                    !countedTracks[$0.trackId, default: false] &&
                    $0.trackletLen > 10
                }.prefix(15) // Only process up to 15 tracks
                
                for track in tracksToCheck {
                    countObject(trackId: track.trackId)
                }
            }
        }
        
        // Clean up tracking data less frequently when we have many tracks
        let cleanupInterval: Int
        if trackedObjects.count > 100 {
            cleanupInterval = 20  // More frequent cleanup with many tracks
        } else if trackedObjects.count > 50 {
            cleanupInterval = 40
        } else {
            cleanupInterval = 60
        }
        
        if frameCount % cleanupInterval == 0 {
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
            
            // Maintain maps at a reasonable size
            if previousPositions.count > 150 {
                let keysToKeep = currentIds.sorted().suffix(100)
                previousPositions = previousPositions.filter { keysToKeep.contains($0.key) }
                historyPositions = historyPositions.filter { keysToKeep.contains($0.key) }
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
