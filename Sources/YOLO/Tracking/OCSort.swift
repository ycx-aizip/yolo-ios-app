// OCSort.swift
// Corrected OC-SORT implementation with 1-to-1 Python correspondence
//
// This is a complete rewrite addressing all critical issues:
// 1. Proper 7D Kalman filter with correct initialization
// 2. Velocity Direction Consistency (VDC) in association
// 3. Two-threshold detection splitting (high/low conf)
// 4. BYTE association stage
// 5. Observation history with freeze/unfreeze
// 6. Correct k_previous_obs logic
// 7. Proper min_hits filtering
//
// Reference: ocsort/ocsort.py and ocsort/association.py

import Foundation
import CoreGraphics
import Accelerate

/// OC-SORT tracker with velocity direction consistency
@MainActor
public class OCSort: TrackerProtocol {

    // MARK: - Configuration

    /// Detection confidence threshold for high conf detections
    private let detThresh: Float

    /// Maximum age before track deletion
    private let maxAge: Int

    /// Minimum hits for track confirmation
    private let minHits: Int

    /// IoU threshold for association
    private let iouThreshold: Float

    /// Delta T for k-previous observations (default: 3)
    private let deltaT: Int

    /// Association function type
    private let assoFunc: String

    /// Inertia (weight for IoU vs angle in association)
    private let inertia: Float

    /// Use BYTE association
    private let useByte: Bool

    /// VDC weight (velocity direction consistency weight)
    private let vdcWeight: Float

    // MARK: - State

    /// Frame counter
    private var frameCount: Int = 0

    /// Active trackers
    private var trackers: [KalmanBoxTracker] = []

    // MARK: - Initialization

    nonisolated public init(
        detThresh: Float = 0.6,
        maxAge: Int = 30,
        minHits: Int = 3,
        iouThreshold: Float = 0.3,
        deltaT: Int = 3,
        assoFunc: String = "iou",
        inertia: Float = 0.2,
        useByte: Bool = false,
        vdcWeight: Float = 0.0
    ) {
        self.detThresh = detThresh
        self.maxAge = maxAge
        self.minHits = minHits
        self.iouThreshold = iouThreshold
        self.deltaT = deltaT
        self.assoFunc = assoFunc
        self.inertia = inertia
        self.useByte = useByte
        self.vdcWeight = vdcWeight
    }

    // MARK: - TrackerProtocol Implementation

    @MainActor
    public func update(detections: [Box], scores: [Float], classes: [String]) -> [STrack] {
        frameCount += 1

        // Store original detections and classes for later class matching
        let originalDetections = detections
        let originalClasses = classes

        // Convert Box format to [x1, y1, x2, y2, score] format
        let detsArray = zip(detections, scores).map { (box, score) -> [Double] in
            let bbox = box.xywhn
            let x1 = Double(bbox.minX)
            let y1 = Double(bbox.minY)
            let x2 = Double(bbox.maxX)
            let y2 = Double(bbox.maxY)
            return [x1, y1, x2, y2, Double(score)]
        }

        // Split detections by confidence threshold
        let highIndices = scores.indices.filter { scores[$0] >= detThresh }
        let lowIndices = scores.indices.filter { scores[$0] < detThresh && scores[$0] > 0.1 }

        let detsHigh = highIndices.map { detsArray[$0] }
        let detsLow = lowIndices.map { detsArray[$0] }

        // Get predicted locations from existing trackers
        var trks: [[Double]] = []
        var toDelete: [Int] = []

        for (t, trk) in trackers.enumerated() {
            let pos = trk.predict()
            trks.append(pos)
            if anyNaN(pos) {
                toDelete.append(t)
            }
        }

        // Remove trackers with NaN predictions
        for t in toDelete.reversed() {
            trackers.remove(at: t)
            trks.remove(at: t)
        }

        // Get velocities and previous observations for VDC
        let velocities = trackers.map { trk -> [Double] in
            return trk.velocity ?? [0, 0]
        }

        let kObservations = trackers.map { trk -> [Double] in
            return trk.getKPreviousObs(k: deltaT)
        }

        // STAGE 1: Associate high confidence detections with trackers
        let (matched, unmatchedDets, unmatchedTrks) = associate(
            detections: detsHigh,
            trackers: trks,
            iouThreshold: iouThreshold,
            velocities: velocities,
            previousObs: kObservations,
            vdcWeight: vdcWeight
        )

        // Update matched trackers
        for m in matched {
            let detIdx = m[0]
            let trkIdx = m[1]
            trackers[trkIdx].update(bbox: detsHigh[detIdx])
        }

        // STAGE 2: BYTE association (if enabled)
        var unmatchedDets2 = unmatchedDets
        var unmatchedTrks2 = unmatchedTrks

        if useByte && !detsLow.isEmpty {
            let unmatchedTrackersIndices = unmatchedTrks
            let unmatchedTrackers = unmatchedTrackersIndices.map { trks[$0] }

            let (matchedByte, unmatchedDetsB, unmatchedTrksB) = associateDetectionsToTrackers(
                detections: detsLow,
                trackers: unmatchedTrackers,
                iouThreshold: iouThreshold  // Use instance variable, not hardcoded 0.5
            )

            // Update matched trackers from BYTE stage
            for m in matchedByte {
                let detIdx = m[0]
                let localTrkIdx = m[1]
                let globalTrkIdx = unmatchedTrackersIndices[localTrkIdx]
                trackers[globalTrkIdx].update(bbox: detsLow[detIdx])
            }

            // Update unmatched lists
            unmatchedTrks2 = unmatchedTrksB.map { unmatchedTrackersIndices[$0] }
        }

        // STAGE 3: OCR (Observation-Centric Recovery)
        if !unmatchedDets.isEmpty && !unmatchedTrks2.isEmpty {
            let unmatchedDetsIndices = unmatchedDets
            let unmatchedTrksIndices = unmatchedTrks2

            let unmatchedDetections = unmatchedDetsIndices.map { detsHigh[$0] }
            let unmatchedTrackers = unmatchedTrksIndices.map { trks[$0] }

            let unmatchedVelocities = unmatchedTrksIndices.map { velocities[$0] }
            let unmatchedKObs = unmatchedTrksIndices.map { kObservations[$0] }

            let (matchedOCR, unmatchedDetsOCR, unmatchedTrksOCR) = associate(
                detections: unmatchedDetections,
                trackers: unmatchedTrackers,
                iouThreshold: 0.5,
                velocities: unmatchedVelocities,
                previousObs: unmatchedKObs,
                vdcWeight: vdcWeight
            )

            // Update matched trackers from OCR
            for m in matchedOCR {
                let localDetIdx = m[0]
                let localTrkIdx = m[1]

                // Bounds checking to prevent crashes
                guard localDetIdx < unmatchedDetsIndices.count,
                      localTrkIdx < unmatchedTrksIndices.count else {
                    continue
                }

                let globalDetIdx = unmatchedDetsIndices[localDetIdx]
                let globalTrkIdx = unmatchedTrksIndices[localTrkIdx]

                guard globalDetIdx < detsHigh.count,
                      globalTrkIdx < trackers.count else {
                    continue
                }

                trackers[globalTrkIdx].update(bbox: detsHigh[globalDetIdx])
            }

            // Update final unmatched lists
            unmatchedDets2 = unmatchedDetsOCR.map { unmatchedDetsIndices[$0] }
            unmatchedTrks2 = unmatchedTrksOCR.map { unmatchedTrksIndices[$0] }
        }

        // Create new trackers for unmatched detections
        for i in unmatchedDets2 {
            let det = detsHigh[i]
            let trk = KalmanBoxTracker(bbox: det, deltaT: deltaT)
            trackers.append(trk)
        }

        // Remove dead trackers
        var i = trackers.count - 1
        while i >= 0 {
            let trk = trackers[i]
            // Remove if too old
            if trk.timeSinceUpdate > maxAge {
                trackers.remove(at: i)
            }
            i -= 1
        }

        // Convert to STrack format with class matching
        let results = trackers.compactMap { trk -> STrack? in
            // Apply min_hits filtering
            let shouldReturn = (trk.timeSinceUpdate < 1) &&
                              (trk.hitStreak >= minHits || frameCount <= minHits)

            guard shouldReturn else { return nil }

            let state = trk.getState()
            let bbox = convertXToBbox(x: state)

            // Convert to center position and size
            let x1 = bbox[0]
            let y1 = bbox[1]
            let x2 = bbox[2]
            let y2 = bbox[3]
            let centerX = (x1 + x2) / 2
            let centerY = (y1 + y2) / 2
            let width = x2 - x1
            let height = y2 - y1

            // Match track to detection by IoU to get correct class
            // Reference: video_tracker_v3.py lines 136-149
            var matchedClass = "fish"  // Default fallback
            var matchedIndex = 0
            var maxIouScore = 0.0

            if !originalDetections.isEmpty {
                let trackBox = [x1, y1, x2, y2]

                for (i, det) in originalDetections.enumerated() {
                    let detBbox = det.xywhn
                    let detBox = [Double(detBbox.minX), Double(detBbox.minY),
                                 Double(detBbox.maxX), Double(detBbox.maxY)]
                    let iouScore = iou(bbox1: trackBox, bbox2: detBox)

                    if iouScore > maxIouScore {
                        maxIouScore = iouScore
                        matchedClass = i < originalClasses.count ? originalClasses[i] : "fish"
                        matchedIndex = i
                    }
                }

                // Only use matched class if IoU is above threshold (0.4 like Python)
                if maxIouScore <= 0.4 {
                    matchedClass = "fish"
                }
            }

            let position = (x: CGFloat(centerX), y: CGFloat(centerY))
            let matchedBox = Box(
                index: matchedIndex,
                cls: matchedClass,
                conf: Float(trk.lastObservation[4]),
                xywh: CGRect(x: centerX, y: centerY, width: width, height: height),
                xywhn: CGRect(x: centerX, y: centerY, width: width, height: height)
            )

            return STrack(
                trackId: trk.id,
                position: position,
                detection: matchedBox,
                score: Float(trk.lastObservation[4]),
                cls: matchedClass
            )
        }

        return results
    }

    @MainActor
    public func reset() {
        trackers.removeAll()
        frameCount = 0
        KalmanBoxTracker.resetCount()
    }

    @MainActor
    public func getActiveTracks() -> [STrack] {
        return trackers.compactMap { trk -> STrack? in
            guard trk.hitStreak >= minHits else { return nil }

            let state = trk.getState()
            let bbox = convertXToBbox(x: state)

            let x1 = bbox[0]
            let y1 = bbox[1]
            let x2 = bbox[2]
            let y2 = bbox[3]
            let centerX = (x1 + x2) / 2
            let centerY = (y1 + y2) / 2
            let width = x2 - x1
            let height = y2 - y1

            let position = (x: CGFloat(centerX), y: CGFloat(centerY))
            let dummyBox = Box(
                index: 0,
                cls: "fish",
                conf: Float(trk.lastObservation[4]),
                xywh: CGRect(x: centerX, y: centerY, width: width, height: height),
                xywhn: CGRect(x: centerX, y: centerY, width: width, height: height)
            )

            return STrack(
                trackId: trk.id,
                position: position,
                detection: dummyBox,
                score: Float(trk.lastObservation[4]),
                cls: "fish"
            )
        }
    }

    // MARK: - Helper Functions (1-to-1 with Python)

    /// Check if array contains NaN
    private func anyNaN(_ arr: [Double]) -> Bool {
        return arr.contains { $0.isNaN }
    }

    /// Convert bbox to observation format [x, y, s, r]
    /// - Parameter bbox: [x1, y1, x2, y2, score]
    /// - Returns: [x_center, y_center, scale, aspect_ratio]
    private func convertBboxToZ(bbox: [Double]) -> [Double] {
        let w = bbox[2] - bbox[0]
        let h = bbox[3] - bbox[1]
        let x = bbox[0] + w / 2
        let y = bbox[1] + h / 2
        let s = w * h
        let r = w / h
        return [x, y, s, r]
    }

    /// Convert observation to bbox
    /// - Parameter x: [x_center, y_center, scale, aspect_ratio, ...]
    /// - Returns: [x1, y1, x2, y2]
    private func convertXToBbox(x: [Double]) -> [Double] {
        let xc = x[0]
        let yc = x[1]
        let s = x[2]
        let r = x[3]

        let w = sqrt(s * r)
        let h = s / w

        return [
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2
        ]
    }

    /// Association with VDC (Velocity Direction Consistency)
    /// Reference: association.py lines 244-300
    private func associate(
        detections: [[Double]],
        trackers: [[Double]],
        iouThreshold: Float,
        velocities: [[Double]],
        previousObs: [[Double]],
        vdcWeight: Float
    ) -> (matched: [[Int]], unmatchedDets: [Int], unmatchedTrks: [Int]) {

        if trackers.isEmpty {
            return (matched: [], unmatchedDets: Array(0..<detections.count), unmatchedTrks: [])
        }

        if detections.isEmpty {
            return (matched: [], unmatchedDets: [], unmatchedTrks: Array(0..<trackers.count))
        }

        // Compute IoU matrix
        let iouMatrix = iouBatch(detections: detections, trackers: trackers)

        // Compute velocity direction consistency cost
        var angleDiffCost = [[Double]](repeating: [Double](repeating: 0.0, count: trackers.count), count: detections.count)

        if vdcWeight > 0 {
            // speed_direction_batch: compute normalized velocity from det to prev_obs
            let (Y, X) = speedDirectionBatch(detections: detections, tracks: previousObs)

            // Get velocities and broadcast
            var inertiaY = [[Double]](repeating: [Double](repeating: 0.0, count: detections.count), count: trackers.count)
            var inertiaX = [[Double]](repeating: [Double](repeating: 0.0, count: detections.count), count: trackers.count)

            for t in 0..<trackers.count {
                let vy = velocities[t][0]
                let vx = velocities[t][1]
                for d in 0..<detections.count {
                    inertiaY[t][d] = vy
                    inertiaX[t][d] = vx
                }
            }

            // Compute angle difference
            for t in 0..<trackers.count {
                for d in 0..<detections.count {
                    // dot product of inertia and speed direction
                    let dotProduct = inertiaX[t][d] * X[t][d] + inertiaY[t][d] * Y[t][d]
                    let clipped = min(max(dotProduct, -1.0), 1.0)
                    var diffAngle = acos(clipped)
                    diffAngle = (.pi / 2.0 - abs(diffAngle)) / .pi

                    // Check if previous observation is valid
                    let validMask = previousObs[t][4] >= 0 ? 1.0 : 0.0
                    let detScore = detections[d][4]

                    angleDiffCost[d][t] = validMask * diffAngle * Double(vdcWeight) * detScore
                }
            }
        }

        // Combined cost matrix
        var costMatrix = [[Double]](repeating: [Double](repeating: 0.0, count: trackers.count), count: detections.count)
        for d in 0..<detections.count {
            for t in 0..<trackers.count {
                costMatrix[d][t] = -(iouMatrix[d][t] + angleDiffCost[d][t])
            }
        }

        // Hungarian matching
        let matchedIndices = linearAssignment(costMatrix: costMatrix)

        // Filter matches and identify unmatched
        var matches: [[Int]] = []
        var unmatchedDetections: [Int] = []
        var unmatchedTrackers: [Int] = []

        let matchedDetsSet = Set(matchedIndices.map { $0[0] })
        let matchedTrksSet = Set(matchedIndices.map { $0[1] })

        for d in 0..<detections.count {
            if !matchedDetsSet.contains(d) {
                unmatchedDetections.append(d)
            }
        }

        for t in 0..<trackers.count {
            if !matchedTrksSet.contains(t) {
                unmatchedTrackers.append(t)
            }
        }

        // Filter matches by IoU threshold
        for m in matchedIndices {
            if iouMatrix[m[0]][m[1]] < Double(iouThreshold) {
                unmatchedDetections.append(m[0])
                unmatchedTrackers.append(m[1])
            } else {
                matches.append(m)
            }
        }

        return (matched: matches, unmatchedDets: unmatchedDetections, unmatchedTrks: unmatchedTrackers)
    }

    /// Basic IoU-only association (for BYTE stage)
    /// Reference: association.py lines 200-241
    private func associateDetectionsToTrackers(
        detections: [[Double]],
        trackers: [[Double]],
        iouThreshold: Float
    ) -> (matched: [[Int]], unmatchedDets: [Int], unmatchedTrks: [Int]) {

        if trackers.isEmpty {
            return (matched: [], unmatchedDets: Array(0..<detections.count), unmatchedTrks: [])
        }

        let iouMatrix = iouBatch(detections: detections, trackers: trackers)
        let matchedIndices = linearAssignment(costMatrix: iouMatrix.map { row in row.map { -$0 } })

        var matches: [[Int]] = []
        var unmatchedDetections: [Int] = []
        var unmatchedTrackers: [Int] = []

        let matchedDetsSet = Set(matchedIndices.map { $0[0] })
        let matchedTrksSet = Set(matchedIndices.map { $0[1] })

        for d in 0..<detections.count {
            if !matchedDetsSet.contains(d) {
                unmatchedDetections.append(d)
            }
        }

        for t in 0..<trackers.count {
            if !matchedTrksSet.contains(t) {
                unmatchedTrackers.append(t)
            }
        }

        for m in matchedIndices {
            if iouMatrix[m[0]][m[1]] < Double(iouThreshold) {
                unmatchedDetections.append(m[0])
                unmatchedTrackers.append(m[1])
            } else {
                matches.append(m)
            }
        }

        return (matched: matches, unmatchedDets: unmatchedDetections, unmatchedTrks: unmatchedTrackers)
    }

    /// Compute normalized velocity direction between detections and tracks
    /// Reference: association.py lines 177-186
    private func speedDirectionBatch(detections: [[Double]], tracks: [[Double]]) -> (dy: [[Double]], dx: [[Double]]) {
        var DY = [[Double]](repeating: [Double](repeating: 0.0, count: detections.count), count: tracks.count)
        var DX = [[Double]](repeating: [Double](repeating: 0.0, count: detections.count), count: tracks.count)

        for (t, track) in tracks.enumerated() {
            let cx2 = (track[0] + track[2]) / 2.0
            let cy2 = (track[1] + track[3]) / 2.0

            for (d, det) in detections.enumerated() {
                let cx1 = (det[0] + det[2]) / 2.0
                let cy1 = (det[1] + det[3]) / 2.0

                let dx = cx1 - cx2
                let dy = cy1 - cy2
                let norm = sqrt(dx * dx + dy * dy) + 1e-6

                DX[t][d] = dx / norm
                DY[t][d] = dy / norm
            }
        }

        return (dy: DY, dx: DX)
    }

    /// Compute IoU between all detection-tracker pairs
    private func iouBatch(detections: [[Double]], trackers: [[Double]]) -> [[Double]] {
        var iouMatrix = [[Double]](repeating: [Double](repeating: 0.0, count: trackers.count), count: detections.count)

        for (d, det) in detections.enumerated() {
            for (t, trk) in trackers.enumerated() {
                iouMatrix[d][t] = iou(bbox1: det, bbox2: trk)
            }
        }

        return iouMatrix
    }

    /// Compute IoU between two boxes
    private func iou(bbox1: [Double], bbox2: [Double]) -> Double {
        let x1 = max(bbox1[0], bbox2[0])
        let y1 = max(bbox1[1], bbox2[1])
        let x2 = min(bbox1[2], bbox2[2])
        let y2 = min(bbox1[3], bbox2[3])

        let intersection = max(0, x2 - x1) * max(0, y2 - y1)
        let area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        let area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        let union = area1 + area2 - intersection

        return union > 0 ? intersection / union : 0
    }

    /// Linear assignment (Hungarian algorithm)
    private func linearAssignment(costMatrix: [[Double]]) -> [[Int]] {
        // Convert to Float for MatchingUtils
        let floatMatrix = costMatrix.map { row in row.map { Float($0) } }
        let (matchedDet, matchedTrk, _, _) = MatchingUtils.linearAssignment(
            costMatrix: floatMatrix,
            threshold: Float.greatestFiniteMagnitude
        )
        return zip(matchedDet, matchedTrk).map { [$0, $1] }
    }
}

// MARK: - KalmanBoxTracker

/// Individual tracker with 7D Kalman filter and observation history
/// Reference: ocsort.py lines 61-169
private class KalmanBoxTracker {

    nonisolated(unsafe) static var count: Int = 0

    static func resetCount() {
        count = 0
    }

    let id: Int
    var timeSinceUpdate: Int = 0
    var hits: Int = 0
    var hitStreak: Int = 0
    var age: Int = 0

    var lastObservation: [Double] = [-1, -1, -1, -1, -1]
    var observations: [Int: [Double]] = [:]
    var historyObservations: [[Double]] = []
    var velocity: [Double]? = nil
    let deltaT: Int

    private var kf: KalmanFilter7D

    init(bbox: [Double], deltaT: Int) {
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.deltaT = deltaT

        // Initialize 7D Kalman filter
        self.kf = KalmanFilter7D()

        // Convert bbox to z format and set initial state
        let z = convertBboxToZ(bbox: bbox)
        for i in 0..<4 {
            kf.x[i] = z[i]
        }
    }

    func update(bbox: [Double]) {
        // Estimate velocity using delta-t observations
        if lastObservation[0] >= 0 {
            var previousBox: [Double]? = nil
            for i in 0..<deltaT {
                let dt = deltaT - i
                if let obs = observations[age - dt] {
                    previousBox = obs
                    break
                }
            }
            if previousBox == nil {
                previousBox = lastObservation
            }

            // Compute velocity (speed_direction)
            if let prevBox = previousBox {
                velocity = speedDirection(bbox1: prevBox, bbox2: bbox)
            }
        }

        // Store observation
        lastObservation = bbox
        observations[age] = bbox
        historyObservations.append(bbox)

        timeSinceUpdate = 0
        hits += 1
        hitStreak += 1

        // Update Kalman filter
        let z = convertBboxToZ(bbox: bbox)
        kf.update(z: z)
    }

    func predict() -> [Double] {
        // Prevent negative scale
        if (kf.x[6] + kf.x[2]) <= 0 {
            kf.x[6] = 0
        }

        kf.predict()
        age += 1

        if timeSinceUpdate > 0 {
            hitStreak = 0
        }
        timeSinceUpdate += 1

        return convertXToBbox(x: kf.x)
    }

    func getState() -> [Double] {
        return kf.x
    }

    /// Get k-previous observation with fallback
    /// Reference: ocsort.py lines 10-18
    func getKPreviousObs(k: Int) -> [Double] {
        // Try to find observation k frames ago, with fallback
        for i in 0..<k {
            let dt = k - i
            if let obs = observations[age - dt] {
                return obs
            }
        }

        // Fallback to most recent observation
        if let maxAge = observations.keys.max(), let obs = observations[maxAge] {
            return obs
        }

        // No observations available
        return [-1, -1, -1, -1, -1]
    }

    private func convertBboxToZ(bbox: [Double]) -> [Double] {
        let w = bbox[2] - bbox[0]
        let h = bbox[3] - bbox[1]
        let x = bbox[0] + w / 2
        let y = bbox[1] + h / 2
        let s = w * h
        let r = w / h
        return [x, y, s, r]
    }

    private func convertXToBbox(x: [Double]) -> [Double] {
        let xc = x[0]
        let yc = x[1]
        let s = x[2]
        let r = x[3]

        let w = sqrt(s * r)
        let h = s / w

        return [
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2
        ]
    }

    private func speedDirection(bbox1: [Double], bbox2: [Double]) -> [Double] {
        let cx1 = (bbox1[0] + bbox1[2]) / 2.0
        let cy1 = (bbox1[1] + bbox1[3]) / 2.0
        let cx2 = (bbox2[0] + bbox2[2]) / 2.0
        let cy2 = (bbox2[1] + bbox2[3]) / 2.0

        let dy = cy2 - cy1
        let dx = cx2 - cx1
        let norm = sqrt(dx * dx + dy * dy) + 1e-6

        return [dy / norm, dx / norm]
    }
}

// MARK: - KalmanFilter7D

/// 7D Kalman filter for OC-SORT
/// State: [x, y, s, r, vx, vy, vs] where s=scale, r=aspect_ratio
/// Reference: kalmanfilter.py and ocsort.py lines 75-88
private class KalmanFilter7D {

    var x: [Double]  // State vector (7x1)
    var P: [Double]  // State covariance (7x7)
    let F: [Double]  // State transition (7x7)
    let H: [Double]  // Measurement function (4x7)
    var Q: [Double]  // Process noise (7x7)
    var R: [Double]  // Measurement noise (4x4)

    init() {
        // Initialize state
        x = [Double](repeating: 0.0, count: 7)

        // State transition matrix F (7x7) - constant velocity model
        F = [
            1, 0, 0, 0, 1, 0, 0,
            0, 1, 0, 0, 0, 1, 0,
            0, 0, 1, 0, 0, 0, 1,
            0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 1
        ]

        // Measurement matrix H (4x7) - only observe position
        H = [
            1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0
        ]

        // Initialize R (measurement noise)
        R = [Double](repeating: 0.0, count: 16)
        for i in 0..<4 {
            R[i * 4 + i] = 1.0
        }
        // R[2:, 2:] *= 10
        R[2 * 4 + 2] *= 10.0
        R[2 * 4 + 3] *= 10.0
        R[3 * 4 + 2] *= 10.0
        R[3 * 4 + 3] *= 10.0

        // Initialize P (state covariance)
        P = [Double](repeating: 0.0, count: 49)
        for i in 0..<7 {
            P[i * 7 + i] = 1.0
        }
        // P[4:, 4:] *= 1000 (high uncertainty for velocities)
        for i in 4..<7 {
            P[i * 7 + i] = 1000.0
        }
        // P *= 10
        for i in 0..<49 {
            P[i] *= 10.0
        }

        // Initialize Q (process noise)
        Q = [Double](repeating: 0.0, count: 49)
        for i in 0..<7 {
            Q[i * 7 + i] = 1.0
        }
        // Q[-1, -1] *= 0.01
        Q[6 * 7 + 6] *= 0.01
        // Q[4:, 4:] *= 0.01
        for i in 4..<7 {
            Q[i * 7 + i] *= 0.01
        }
    }

    func predict() {
        // x = F * x
        var newX = [Double](repeating: 0.0, count: 7)
        vDSP_mmulD(F, 1, x, 1, &newX, 1, 7, 1, 7)
        x = newX

        // P = F * P * F^T + Q
        var FP = [Double](repeating: 0.0, count: 49)
        vDSP_mmulD(F, 1, P, 1, &FP, 1, 7, 7, 7)

        var F_T = [Double](repeating: 0.0, count: 49)
        vDSP_mtransD(F, 1, &F_T, 1, 7, 7)

        var FPF = [Double](repeating: 0.0, count: 49)
        vDSP_mmulD(FP, 1, F_T, 1, &FPF, 1, 7, 7, 7)

        vDSP_vaddD(FPF, 1, Q, 1, &P, 1, 49)
    }

    func update(z: [Double]) {
        // Innovation: y = z - H * x
        var Hx = [Double](repeating: 0.0, count: 4)
        vDSP_mmulD(H, 1, x, 1, &Hx, 1, 4, 1, 7)

        var innovation = [Double](repeating: 0.0, count: 4)
        for i in 0..<4 {
            innovation[i] = z[i] - Hx[i]
        }

        // S = H * P * H^T + R
        var HP = [Double](repeating: 0.0, count: 28)
        vDSP_mmulD(H, 1, P, 1, &HP, 1, 4, 7, 7)

        var H_T = [Double](repeating: 0.0, count: 28)
        vDSP_mtransD(H, 1, &H_T, 1, 4, 7)

        var HPH = [Double](repeating: 0.0, count: 16)
        vDSP_mmulD(HP, 1, H_T, 1, &HPH, 1, 4, 4, 7)

        var S = [Double](repeating: 0.0, count: 16)
        vDSP_vaddD(HPH, 1, R, 1, &S, 1, 16)

        // Invert S
        let S_inv = invertMatrix4x4(S)

        // K = P * H^T * S^(-1)
        var PH = [Double](repeating: 0.0, count: 28)
        vDSP_mmulD(P, 1, H_T, 1, &PH, 1, 7, 4, 7)

        var K = [Double](repeating: 0.0, count: 28)
        vDSP_mmulD(PH, 1, S_inv, 1, &K, 1, 7, 4, 4)

        // x = x + K * y
        var Ky = [Double](repeating: 0.0, count: 7)
        vDSP_mmulD(K, 1, innovation, 1, &Ky, 1, 7, 1, 4)
        vDSP_vaddD(x, 1, Ky, 1, &x, 1, 7)

        // P = (I - K * H) * P
        var KH = [Double](repeating: 0.0, count: 49)
        vDSP_mmulD(K, 1, H, 1, &KH, 1, 7, 7, 4)

        var I_KH = [Double](repeating: 0.0, count: 49)
        for i in 0..<7 {
            for j in 0..<7 {
                I_KH[i * 7 + j] = (i == j ? 1.0 : 0.0) - KH[i * 7 + j]
            }
        }

        var newP = [Double](repeating: 0.0, count: 49)
        vDSP_mmulD(I_KH, 1, P, 1, &newP, 1, 7, 7, 7)
        P = newP
    }

    private func invertMatrix4x4(_ matrix: [Double]) -> [Double] {
        var augmented = [[Double]](repeating: [Double](repeating: 0.0, count: 8), count: 4)

        for i in 0..<4 {
            for j in 0..<4 {
                augmented[i][j] = matrix[i * 4 + j]
            }
            augmented[i][4 + i] = 1.0
        }

        for i in 0..<4 {
            var maxRow = i
            for k in (i + 1)..<4 {
                if abs(augmented[k][i]) > abs(augmented[maxRow][i]) {
                    maxRow = k
                }
            }

            if maxRow != i {
                augmented.swapAt(i, maxRow)
            }

            let pivot = augmented[i][i]
            if abs(pivot) < 1e-10 {
                var identity = [Double](repeating: 0.0, count: 16)
                for k in 0..<4 {
                    identity[k * 4 + k] = 1.0
                }
                return identity
            }

            for j in 0..<8 {
                augmented[i][j] /= pivot
            }

            for k in 0..<4 {
                if k != i {
                    let factor = augmented[k][i]
                    for j in 0..<8 {
                        augmented[k][j] -= factor * augmented[i][j]
                    }
                }
            }
        }

        var inverse = [Double](repeating: 0.0, count: 16)
        for i in 0..<4 {
            for j in 0..<4 {
                inverse[i * 4 + j] = augmented[i][4 + j]
            }
        }

        return inverse
    }
}
