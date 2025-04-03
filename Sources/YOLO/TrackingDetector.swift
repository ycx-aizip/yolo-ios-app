// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
//  This file is part of the Ultralytics YOLO Package, implementing object detection with tracking.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//
//  The TrackingDetector class extends ObjectDetector to add object tracking capabilities.
//  It uses ByteTrack to maintain consistent object IDs across frames and supports fish counting.

import Foundation
import UIKit
import Vision

/// Specialized predictor for object detection with tracking capabilities
class TrackingDetector: ObjectDetector {
    /// ByteTracker for object tracking
    private let tracker = ByteTracker()
    
    /// List of tracks from the previous frame
    private var tracks: [STrack] = []
    
    /// Callback for when count changes
    var onCountChanged: ((Int) -> Void)?
    
    /// Fish counting properties
    private var fishCount: Int = 0
    private var fishPassingInfo: [Int: (crossed: Bool, direction: String?)] = [:]
    
    /// Threshold values (normalized 0-1)
    var upperThreshold: CGFloat = 0.3
    var bottomThreshold: CGFloat = 0.7
    var leftThreshold: CGFloat = 0.3
    var rightThreshold: CGFloat = 0.7
    
    /// Active thresholds based on camera angle
    var activeThresholds: Set<String> = ["upper", "bottom", "left", "right"]
    
    /// Factory method to create a TrackingDetector instance
    /// - Parameters:
    ///   - unwrappedModelURL: URL to the CoreML model
    ///   - isRealTime: Whether the detector will be used for real-time processing
    ///   - completion: Completion handler called with the created detector or an error
    // public static func create(
    //     unwrappedModelURL: URL,
    //     isRealTime: Bool = false,
    //     completion: @escaping (Result<TrackingDetector, Error>) -> Void
    // ) {
    //     // Delegate to ObjectDetector.create() but cast the result to TrackingDetector
    //     ObjectDetector.create(unwrappedModelURL: unwrappedModelURL, isRealTime: isRealTime) { result in
    //         switch result {
    //         case .success(let predictor):
    //             // Create a new TrackingDetector and copy properties from the ObjectDetector
    //             let trackingDetector = TrackingDetector()
                
    //             // Copy necessary properties from the ObjectDetector
    //             trackingDetector.detector = predictor.detector
    //             trackingDetector.visionRequest = predictor.visionRequest
    //             trackingDetector.labels = predictor.labels
    //             trackingDetector.inputSize = predictor.inputSize
    //             trackingDetector.modelInputSize = predictor.modelInputSize
    //             trackingDetector.isModelLoaded = predictor.isModelLoaded
                
    //             print("DEBUG: Created TrackingDetector with \(trackingDetector.labels.count) classes")
                
    //             // Return the TrackingDetector
    //             completion(.success(trackingDetector))
                
    //         case .failure(let error):
    //             completion(.failure(error))
    //         }
    //     }
    // }
    
    /// Override the process observations method to add tracking
    override func processObservations(for request: VNRequest, error: Error?) {
        // First, let the parent class process the basic detections
        super.processObservations(for: request, error: error)
        
        // Then, extract the bounding boxes from the most recent result
        if let onResultsListener = self.currentOnResultsListener {
            // Get the detection boxes from the most recent result
            if let results = request.results as? [VNRecognizedObjectObservation] {
                var boxes: [CGRect] = []
                var scores: [Float] = []
                var classIds: [Int] = []
                
                for i in 0..<min(results.count, 100) {
                    let prediction = results[i]
                    let invertedBox = CGRect(
                        x: prediction.boundingBox.minX, 
                        y: 1 - prediction.boundingBox.maxY,
                        width: prediction.boundingBox.width, 
                        height: prediction.boundingBox.height)
                    let imageRect = VNImageRectForNormalizedRect(
                        invertedBox, Int(inputSize.width), Int(inputSize.height))
                    
                    boxes.append(imageRect)
                    scores.append(prediction.labels[0].confidence)
                    
                    let label = prediction.labels[0].identifier
                    let index = self.labels.firstIndex(of: label) ?? 0
                    classIds.append(index)
                }
                
                // Update tracks with ByteTracker
                if !boxes.isEmpty {
                    self.tracks = tracker.update(detections: boxes, scores: scores, classIds: classIds)
                    print("DEBUG: TrackingDetector has \(tracks.count) active tracks")
                    
                    // Process tracks for fish counting
                    processTracksForCounting()
                }
            }
        }
    }
    
    /// Process tracks for fish counting
    private func processTracksForCounting() {
        // Skip if no active thresholds
        if activeThresholds.isEmpty {
            return
        }
        
        // For each active track, check if it crossed a threshold
        for track in tracks {
            let trackId = track.trackId
            let trackRect = track.bbox
            
            // Center point of the track
            let centerX = trackRect.midX / inputSize.width
            let centerY = trackRect.midY / inputSize.height
            
            // Check if this is a new track
            if fishPassingInfo[trackId] == nil {
                fishPassingInfo[trackId] = (crossed: false, direction: nil)
            }
            
            // Check upper threshold crossing
            if activeThresholds.contains("upper") && !fishPassingInfo[trackId]!.crossed {
                if centerY < upperThreshold {
                    // Track crossed upper threshold
                    fishPassingInfo[trackId] = (crossed: true, direction: "up")
                    fishCount += 1
                    print("DEBUG: Fish #\(trackId) crossed upper threshold, count = \(fishCount)")
                    notifyCountChanged()
                }
            }
            
            // Check bottom threshold crossing
            if activeThresholds.contains("bottom") && !fishPassingInfo[trackId]!.crossed {
                if centerY > bottomThreshold {
                    // Track crossed bottom threshold
                    fishPassingInfo[trackId] = (crossed: true, direction: "down")
                    fishCount += 1
                    print("DEBUG: Fish #\(trackId) crossed bottom threshold, count = \(fishCount)")
                    notifyCountChanged()
                }
            }
            
            // Check left threshold crossing
            if activeThresholds.contains("left") && !fishPassingInfo[trackId]!.crossed {
                if centerX < leftThreshold {
                    // Track crossed left threshold
                    fishPassingInfo[trackId] = (crossed: true, direction: "left")
                    fishCount += 1
                    print("DEBUG: Fish #\(trackId) crossed left threshold, count = \(fishCount)")
                    notifyCountChanged()
                }
            }
            
            // Check right threshold crossing
            if activeThresholds.contains("right") && !fishPassingInfo[trackId]!.crossed {
                if centerX > rightThreshold {
                    // Track crossed right threshold
                    fishPassingInfo[trackId] = (crossed: true, direction: "right")
                    fishCount += 1
                    print("DEBUG: Fish #\(trackId) crossed right threshold, count = \(fishCount)")
                    notifyCountChanged()
                }
            }
        }
        
        // Clean up old tracks
        cleanupOldTracks()
    }
    
    /// Clean up tracks that are no longer active
    private func cleanupOldTracks() {
        let activeTrackIds = Set(tracks.map { $0.trackId })
        let inactiveTrackIds = Set(fishPassingInfo.keys).subtracting(activeTrackIds)
        
        for trackId in inactiveTrackIds {
            fishPassingInfo.removeValue(forKey: trackId)
        }
    }
    
    /// Reset the fish count
    func resetCount() {
        fishCount = 0
        fishPassingInfo.removeAll()
        notifyCountChanged()
    }
    
    /// Get the current fish count
    var currentCount: Int {
        return fishCount
    }
    
    /// Configure active thresholds based on camera angle
    func configureForCameraAngle(_ angle: String) {
        // Reset active thresholds
        activeThresholds.removeAll()
        
        // Set active thresholds based on camera angle
        switch angle {
        case "Front View":
            activeThresholds.insert("upper")
            activeThresholds.insert("bottom")
            activeThresholds.insert("left")
            activeThresholds.insert("right")
        case "Top View", "Bottom View":
            activeThresholds.insert("upper")
            activeThresholds.insert("bottom")
        case "Left View":
            activeThresholds.insert("right")
        case "Right View":
            activeThresholds.insert("left")
        default:
            // Default to all thresholds
            activeThresholds.insert("upper")
            activeThresholds.insert("bottom")
            activeThresholds.insert("left")
            activeThresholds.insert("right")
        }
        
        // Reset count when changing angles
        resetCount()
    }
    
    /// Notify when count changes
    private func notifyCountChanged() {
        // Ensure UI updates happen on the main thread
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.onCountChanged?(self.fishCount)
        }
    }
} 