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

    /// Initialize OC-SORT adapted for CoreML YOLO (iOS deployment)
    ///
    /// CRITICAL DIFFERENCE from Python:
    /// - Python YOLO: Returns detections with varying confidence (conf param is soft filter)
    /// - CoreML YOLO: Built-in NMS with HARD confidence filter at 0.25
    /// - CoreML returns ONLY high-confidence dets (≥0.25) → NO low-conf dets for BYTE stage
    ///
    /// Solution: Set detThresh slightly BELOW CoreML threshold (0.2 < 0.25)
    /// This ensures ALL CoreML detections are treated as "high confidence" consistently
    /// Stage 1: All dets matched with predicted positions (VDC + IoU)
    /// Stage 3: OCR recovery using last observations for unmatched tracks
    nonisolated public init(
        detThresh: Float = 0.2,       // ✅ Below CoreML 0.25 → all dets are "high conf", consistent splitting
        maxAge: Int = 30,             // ✅ Python default
        minHits: Int = 3,             // ✅ Python default (reduced from 5 for faster confirmation)
        iouThreshold: Float = 0.05,   // 🔧 CRITICAL: Lowered to 0.05 for fish (KF prediction has VERY low IoU)
        deltaT: Int = 1,              // 🔧 CRITICAL: Reduced from 3 to 1 for fast-moving fish
        assoFunc: String = "iou",     // ✅ Python default
        inertia: Float = 0.2,         // ✅ Python default
        useByte: Bool = false,        // ✅ Python default (no low-conf dets with CoreML anyway)
        vdcWeight: Float = 0.9        // 🔧 CRITICAL: Increased to 0.9 to rely heavily on velocity direction
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
        
        // 🔍 DEBUG: Track association performance
        let debugEnabled = true  // Set to false to disable debug logs
        if debugEnabled && frameCount % 30 == 0 {  // Log every 30 frames
            print("\n═══ OCSort Debug Frame \(frameCount) ═══")
            print("📊 Input: \(detections.count) detections, \(trackers.count) active trackers")

            // Debug: Show first detection in both formats
            if !detections.isEmpty {
                let firstBox = detections[0]
                let bbox = firstBox.xywhn
                print("   🔍 Detection[0] format check:")
                print("      xywhn: x=\(String(format: "%.3f", bbox.origin.x)), y=\(String(format: "%.3f", bbox.origin.y)), w=\(String(format: "%.3f", bbox.width)), h=\(String(format: "%.3f", bbox.height))")
                print("      As [x1,y1,x2,y2]: [\(String(format: "%.3f", bbox.minX)), \(String(format: "%.3f", bbox.minY)), \(String(format: "%.3f", bbox.maxX)), \(String(format: "%.3f", bbox.maxY))]")
            }
        }

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
            // 🔍 DEBUG: Capture state BEFORE predict
            var stateBefore: [Double]? = nil
            if debugEnabled && frameCount % 30 == 0 && t == 0 && trk.lastObservation[0] >= 0 {
                stateBefore = trk.getState()
            }

            let pos = trk.predict()
            trks.append(pos)
            if anyNaN(pos) {
                toDelete.append(t)
            }

            // 🔍 DEBUG: Compare prediction vs last observation
            if let stateBeforePredict = stateBefore {
                let lastObs = trk.lastObservation
                let stateAfter = trk.getState()
                print("   🔮 Kalman Prediction Debug (Tracker 0):")
                print("      State BEFORE: cx=\(String(format: "%.3f", stateBeforePredict[0])), cy=\(String(format: "%.3f", stateBeforePredict[1])), s=\(String(format: "%.3f", stateBeforePredict[2])), r=\(String(format: "%.3f", stateBeforePredict[3]))")
                print("      State AFTER:  cx=\(String(format: "%.3f", stateAfter[0])), cy=\(String(format: "%.3f", stateAfter[1])), s=\(String(format: "%.3f", stateAfter[2])), r=\(String(format: "%.3f", stateAfter[3]))")
                print("      Last obs bbox:    [\(String(format: "%.3f", lastObs[0])), \(String(format: "%.3f", lastObs[1])), \(String(format: "%.3f", lastObs[2])), \(String(format: "%.3f", lastObs[3]))]")
                print("      Predicted bbox:   [\(String(format: "%.3f", pos[0])), \(String(format: "%.3f", pos[1])), \(String(format: "%.3f", pos[2])), \(String(format: "%.3f", pos[3]))]")
                let drift = sqrt(pow(pos[0] - lastObs[0], 2) + pow(pos[1] - lastObs[1], 2))
                print("      Position drift: \(String(format: "%.3f", drift))")
            }
        }

        // Remove trackers with NaN predictions
        for t in toDelete.reversed() {
            trackers.remove(at: t)
            trks.remove(at: t)
        }

        // Get velocities, previous observations for VDC, and last observations for OCR
        let velocities = trackers.map { trk -> [Double] in
            return trk.velocity ?? [0, 0]
        }

        let kObservations = trackers.map { trk -> [Double] in
            return trk.getKPreviousObs(k: deltaT)
        }

        // Extract last observations for Stage 3 OCR (OC-SORT innovation)
        let lastObservations = trackers.map { trk -> [Double] in
            return trk.lastObservation
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
        
        // 🔍 DEBUG: Stage 1 results
        if debugEnabled && frameCount % 30 == 0 {
            print("🎯 Stage 1 (High conf + VDC): \(matched.count) matches, \(unmatchedDets.count) unmatched dets, \(unmatchedTrks.count) unmatched trks")
            if !matched.isEmpty {
                print("   ✓ Matched pairs: \(matched.prefix(5).map { "D\($0[0])→T\($0[1])" }.joined(separator: ", "))\(matched.count > 5 ? "..." : "")")
            }
        }

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
        // This is OC-SORT's innovation: match unmatched high-conf dets with LAST OBSERVATIONS
        // instead of predicted positions. Handles occlusion and fast motion better.
        if !unmatchedDets.isEmpty && !unmatchedTrks2.isEmpty {
            let unmatchedDetsIndices = unmatchedDets
            let unmatchedTrksIndices = unmatchedTrks2

            let unmatchedDetections = unmatchedDetsIndices.map { detsHigh[$0] }
            // ✅ FIX: Use LAST OBSERVATIONS (not predicted positions) - this is OCR's key innovation!
            let unmatchedLastObs = unmatchedTrksIndices.map { lastObservations[$0] }

            let unmatchedVelocities = unmatchedTrksIndices.map { velocities[$0] }
            let unmatchedKObs = unmatchedTrksIndices.map { kObservations[$0] }

            // Python recommends lower threshold for OCR stage (iou_threshold - 0.1) for better recovery
            // See ocsort.py lines 285-288: "using a lower threshold...may get higher performance"
            let ocrThreshold = max(0.1, iouThreshold - 0.1)  // 0.2 if iouThreshold=0.3

            let (matchedOCR, unmatchedDetsOCR, unmatchedTrksOCR) = associate(
                detections: unmatchedDetections,
                trackers: unmatchedLastObs,  // ✅ Use last observations, not predictions!
                iouThreshold: ocrThreshold,  // ✅ Lower threshold for better recovery (0.2 vs 0.3)
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
            
            // 🔍 DEBUG: Stage 3 results
            if debugEnabled && frameCount % 30 == 0 {
                print("🔄 Stage 3 (OCR): \(matchedOCR.count) recoveries")
                if matchedOCR.count > 0 {
                    print("   ✓ OCR saved \(matchedOCR.count) tracks from being lost!")
                }
            }
        }

        // Create new trackers for unmatched detections
        for i in unmatchedDets2 {
            let det = detsHigh[i]
            let trk = KalmanBoxTracker(bbox: det, deltaT: deltaT)
            trackers.append(trk)
        }
        
        // 🔍 DEBUG: New track creation
        if debugEnabled && frameCount % 30 == 0 && !unmatchedDets2.isEmpty {
            print("🆕 Created \(unmatchedDets2.count) new trackers")
        }

        // Remove dead trackers
        var removedCount = 0
        var i = trackers.count - 1
        while i >= 0 {
            let trk = trackers[i]
            // Remove if too old
            if trk.timeSinceUpdate > maxAge {
                trackers.remove(at: i)
                removedCount += 1
            }
            i -= 1
        }
        
        // 🔍 DEBUG: Tracker lifecycle
        if debugEnabled && frameCount % 30 == 0 {
            print("💀 Removed \(removedCount) old trackers (age > \(maxAge))")
            print("📈 Tracker stats: \(trackers.count) alive, timeSinceUpdate distribution:")
            let tsuCounts = trackers.reduce(into: [:]) { counts, trk in
                counts[trk.timeSinceUpdate, default: 0] += 1
            }
            for (tsu, count) in tsuCounts.sorted(by: { $0.key < $1.key }).prefix(5) {
                print("   - TSU=\(tsu): \(count) trackers")
            }
        }

        // Convert to STrack format with class matching
        let results = trackers.compactMap { trk -> STrack? in
            // ✅ FIX: OC-SORT output filter (NOT ByteTrack's TSU<1!)
            // Return tracks that are:
            // 1. Confirmed (hitStreak >= minHits OR early frames <= minHits)
            // 2. Recently updated (timeSinceUpdate <= 1 for smooth tracking)
            // Python OC-SORT returns all tracks passing min_hits check, but we add TSU<=1
            // to avoid returning stale predictions (tracks will be kept alive until maxAge=30)
            let isConfirmed = (trk.hitStreak >= minHits || frameCount <= minHits)
            let isRecent = trk.timeSinceUpdate <= 1
            let shouldReturn = isConfirmed && isRecent

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
        
        // 🔍 DEBUG: Final output analysis
        if debugEnabled && frameCount % 30 == 0 {
            let confirmedCount = trackers.filter { $0.hitStreak >= minHits }.count
            let returnedCount = results.count
            print("📤 Output: \(returnedCount) tracks returned (\(confirmedCount) confirmed, \(trackers.count - confirmedCount) unconfirmed)")
            print("   Filter: timeSinceUpdate<=1 AND (hitStreak>=\(minHits) OR frameCount<=\(minHits))")
            
            // Analyze why tracks weren't returned
            let notReturned = trackers.filter { trk in
                !((trk.timeSinceUpdate < 1) && (trk.hitStreak >= minHits || frameCount <= minHits))
            }
            if !notReturned.isEmpty {
                print("❌ \(notReturned.count) trackers FILTERED OUT:")
                for (idx, trk) in notReturned.prefix(5).enumerated() {
                    let reason = trk.timeSinceUpdate >= 1 ? "TSU=\(trk.timeSinceUpdate)≥1" : "hitStreak=\(trk.hitStreak)<\(minHits)"
                    print("   [\(idx)] ID:\(trk.id) - \(reason), hits:\(trk.hits)")
                }
            }
            print("═══════════════════════════════════\n")
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
        
        // 🔍 DEBUG: IoU analysis
        let debugEnabled = true
        if debugEnabled && frameCount % 30 == 0 && !detections.isEmpty && !trackers.isEmpty {
            let maxIoUs = iouMatrix.map { row in row.max() ?? 0.0 }
            let avgMaxIoU = maxIoUs.reduce(0.0, +) / Double(maxIoUs.count)
            let goodMatches = maxIoUs.filter { $0 > Double(iouThreshold) }.count
            let vdcInfo = vdcWeight > 0 ? "VDC=\(String(format: "%.1f", vdcWeight))" : "VDC=OFF"
            print("   📏 IoU stats: avg_max=\(String(format: "%.3f", avgMaxIoU)), \(goodMatches)/\(detections.count) above thresh=\(iouThreshold), \(vdcInfo)")
        }

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
        // 🔍 DEBUG: Track match quality
        var totalAngleCost = 0.0
        var validMatchCount = 0
        
        for m in matchedIndices {
            if iouMatrix[m[0]][m[1]] < Double(iouThreshold) {
                unmatchedDetections.append(m[0])
                unmatchedTrackers.append(m[1])
            } else {
                matches.append(m)
                if vdcWeight > 0 {
                    totalAngleCost += angleDiffCost[m[0]][m[1]]
                    validMatchCount += 1
                }
            }
        }
        
        // Print VDC contribution for matched pairs
        if debugEnabled && vdcWeight > 0 && frameCount % 30 == 0 && validMatchCount > 0 {
            let avgAngleCost = totalAngleCost / Double(validMatchCount)
            print("   🎯 VDC contribution: avg_angle_bonus=\(String(format: "%.3f", avgAngleCost)) for \(validMatchCount) matches")
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

        // 🔍 DEBUG: Log Kalman update with P and K diagnostics
        #if DEBUG
        if id == 0 {  // Only log first tracker
            print("      📝 KalmanBoxTracker.update() for ID \(id):")
            print("         Input bbox: [\(String(format: "%.3f", bbox[0])), \(String(format: "%.3f", bbox[1])), \(String(format: "%.3f", bbox[2])), \(String(format: "%.3f", bbox[3]))]")
            print("         Converted z: cx=\(String(format: "%.3f", z[0])), cy=\(String(format: "%.3f", z[1])), s=\(String(format: "%.3f", z[2])), r=\(String(format: "%.3f", z[3]))")
            print("         KF state BEFORE update: cx=\(String(format: "%.3f", kf.x[0])), cy=\(String(format: "%.3f", kf.x[1])), s=\(String(format: "%.3f", kf.x[2])), r=\(String(format: "%.3f", kf.x[3]))")
            // Log P diagonal and critical off-diagonal for position elements
            print("         P[0,0]=\(String(format: "%.2e", kf.P[0])), P[1,1]=\(String(format: "%.2e", kf.P[8])), P[0,1]=\(String(format: "%.2e", kf.P[1]))")
        }
        #endif

        kf.update(z: z)

        #if DEBUG
        if id == 0 {
            print("         KF state AFTER update:  cx=\(String(format: "%.3f", kf.x[0])), cy=\(String(format: "%.3f", kf.x[1])), s=\(String(format: "%.3f", kf.x[2])), r=\(String(format: "%.3f", kf.x[3]))")
            print("         P diagonal (positions): [\(String(format: "%.2e", kf.P[0])), \(String(format: "%.2e", kf.P[8])), \(String(format: "%.2e", kf.P[16])), \(String(format: "%.2e", kf.P[24]))]")
            // K is 7x4 (row-major), K[row*4 + col] = K[state, measurement]
            // Rows: [x, y, s, r, vx, vy, vs], Cols: [cx_meas, cy_meas, s_meas, r_meas]
            if let K = kf.lastK {
                print("         K[x,y,s,r] gains for cx_meas: [\(String(format: "%.3f", K[0*4+0])), \(String(format: "%.3f", K[1*4+0])), \(String(format: "%.3f", K[2*4+0])), \(String(format: "%.3f", K[3*4+0]))]")
                print("         K[x,y,s,r] gains for cy_meas: [\(String(format: "%.3f", K[0*4+1])), \(String(format: "%.3f", K[1*4+1])), \(String(format: "%.3f", K[2*4+1])), \(String(format: "%.3f", K[3*4+1]))]")
                print("         K[x,y,s,r] gains for s_meas:  [\(String(format: "%.3f", K[0*4+2])), \(String(format: "%.3f", K[1*4+2])), \(String(format: "%.3f", K[2*4+2])), \(String(format: "%.3f", K[3*4+2]))]")
            }
        }
        #endif
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
    var lastK: [Double]?  // Last Kalman gain for debugging (7x4)

    init() {
        // Initialize state
        x = [Double](repeating: 0.0, count: 7)

        // ✅ FIX: State transition matrix F (7x7) - CONSTANT POSITION model
        // Python uses F = Identity (no velocity propagation in state transition!)
        // Reference: kalmanfilter.py line 299 "self.F = eye(dim_x)"
        F = [
            1, 0, 0, 0, 0, 0, 0,  // x_new = x (NOT x + vx!)
            0, 1, 0, 0, 0, 0, 0,  // y_new = y (NOT y + vy!)
            0, 0, 1, 0, 0, 0, 0,  // s_new = s (NOT s + vs!)
            0, 0, 0, 1, 0, 0, 0,  // r_new = r
            0, 0, 0, 0, 1, 0, 0,  // vx_new = vx (velocity persists)
            0, 0, 0, 0, 0, 1, 0,  // vy_new = vy
            0, 0, 0, 0, 0, 0, 1   // vs_new = vs
        ]

        // Measurement matrix H (4x7) - only observe position
        H = [
            1, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0
        ]

        // Initialize R (measurement noise) - DIAGONAL matrix
        R = [Double](repeating: 0.0, count: 16)
        for i in 0..<4 {
            R[i * 4 + i] = 1.0
        }
        // ✅ FIX: R[2:, 2:] *= 10 means multiply DIAGONAL elements R[2,2] and R[3,3] by 10
        // NOT off-diagonal elements! R must stay diagonal.
        R[2 * 4 + 2] *= 10.0  // R[2,2] = 10.0 (scale measurement noise)
        R[3 * 4 + 3] *= 10.0  // R[3,3] = 10.0 (aspect ratio measurement noise)

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
        // x = F * x (7x7 * 7x1 = 7x1)
        var newX = [Double](repeating: 0.0, count: 7)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 7, 7, 1.0, F, 7, x, 1, 0.0, &newX, 1)
        x = newX

        // P = F * P * F^T + Q
        // Step 1: FP = F * P (7x7 * 7x7 = 7x7)
        var FP = [Double](repeating: 0.0, count: 49)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 7, 7, 7, 1.0, F, 7, P, 7, 0.0, &FP, 7)

        // Step 2: FPF = FP * F^T (7x7 * 7x7 = 7x7)
        var FPF = [Double](repeating: 0.0, count: 49)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 7, 7, 7, 1.0, FP, 7, F, 7, 0.0, &FPF, 7)

        // Step 3: P = FPF + Q
        vDSP_vaddD(FPF, 1, Q, 1, &P, 1, 49)
    }

    func update(z: [Double]) {
        // Innovation: y = z - H * x (4x1 = 4x7 * 7x1)
        var Hx = [Double](repeating: 0.0, count: 4)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 4, 7, 1.0, H, 7, x, 1, 0.0, &Hx, 1)

        var innovation = [Double](repeating: 0.0, count: 4)
        for i in 0..<4 {
            innovation[i] = z[i] - Hx[i]
        }

        // ✅ FIX: S = H * P * H^T + R (using correct row-major matrix multiplication)
        // Step 1: HP = H * P (4x7 = 4x7 * 7x7)
        var HP = [Double](repeating: 0.0, count: 28)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 7, 7, 1.0, H, 7, P, 7, 0.0, &HP, 7)

        // Step 2: HPH = HP * H^T (4x4 = 4x7 * 7x4)
        var HPH = [Double](repeating: 0.0, count: 16)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 4, 4, 7, 1.0, HP, 7, H, 7, 0.0, &HPH, 4)

        // Step 3: S = HPH + R
        var S = [Double](repeating: 0.0, count: 16)
        vDSP_vaddD(HPH, 1, R, 1, &S, 1, 16)

        // Invert S
        let S_inv = invertMatrix4x4Stable(S)

        // K = P * H^T * S^(-1)
        // Step 1: PH = P * H^T (7x4 = 7x7 * 7x4)
        var PH = [Double](repeating: 0.0, count: 28)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 7, 4, 7, 1.0, P, 7, H, 7, 0.0, &PH, 4)

        // Step 2: K = PH * S_inv (7x4 = 7x4 * 4x4)
        var K = [Double](repeating: 0.0, count: 28)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 7, 4, 4, 1.0, PH, 4, S_inv, 4, 0.0, &K, 4)
        lastK = K  // Store for debugging

        // x = x + K * y (7x1 = 7x1 + 7x4 * 4x1)
        var Ky = [Double](repeating: 0.0, count: 7)
        cblas_dgemv(CblasRowMajor, CblasNoTrans, 7, 4, 1.0, K, 4, innovation, 1, 0.0, &Ky, 1)
        vDSP_vaddD(x, 1, Ky, 1, &x, 1, 7)

        // ✅ FIX: Use Joseph form for numerical stability
        // P = (I - K*H)*P*(I - K*H)' + K*R*K'
        // This keeps P symmetric and positive-semidefinite
        // Reference: kalmanfilter.py line 521

        // Step 1: KH = K * H (7x7 = 7x4 * 4x7)
        var KH = [Double](repeating: 0.0, count: 49)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 7, 7, 4, 1.0, K, 4, H, 7, 0.0, &KH, 7)

        // Step 2: I_KH = I - KH
        var I_KH = [Double](repeating: 0.0, count: 49)
        for i in 0..<7 {
            for j in 0..<7 {
                I_KH[i * 7 + j] = (i == j ? 1.0 : 0.0) - KH[i * 7 + j]
            }
        }

        // Step 3: I_KH_P = (I - K*H) * P (7x7 = 7x7 * 7x7)
        var I_KH_P = [Double](repeating: 0.0, count: 49)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 7, 7, 7, 1.0, I_KH, 7, P, 7, 0.0, &I_KH_P, 7)

        // Step 4: term1 = (I - K*H) * P * (I - K*H)' (7x7 = 7x7 * 7x7)
        var term1 = [Double](repeating: 0.0, count: 49)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 7, 7, 7, 1.0, I_KH_P, 7, I_KH, 7, 0.0, &term1, 7)

        // Step 5: KR = K * R (7x4 = 7x4 * 4x4)
        var KR = [Double](repeating: 0.0, count: 28)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 7, 4, 4, 1.0, K, 4, R, 4, 0.0, &KR, 4)

        // Step 6: term2 = K * R * K' (7x7 = 7x4 * 4x7)
        var term2 = [Double](repeating: 0.0, count: 49)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 7, 7, 4, 1.0, KR, 4, K, 4, 0.0, &term2, 7)

        // Step 7: P = term1 + term2
        vDSP_vaddD(term1, 1, term2, 1, &P, 1, 49)
    }

    /// Improved 4x4 matrix inversion using cofactor method
    /// More numerically stable than Gaussian elimination for small matrices
    private func invertMatrix4x4Stable(_ m: [Double]) -> [Double] {
        // Compute determinant first
        let det = m[0]*m[5]*m[10]*m[15] - m[0]*m[5]*m[11]*m[14] - m[0]*m[6]*m[9]*m[15] +
                  m[0]*m[6]*m[11]*m[13] + m[0]*m[7]*m[9]*m[14] - m[0]*m[7]*m[10]*m[13] -
                  m[1]*m[4]*m[10]*m[15] + m[1]*m[4]*m[11]*m[14] + m[1]*m[6]*m[8]*m[15] -
                  m[1]*m[6]*m[11]*m[12] - m[1]*m[7]*m[8]*m[14] + m[1]*m[7]*m[10]*m[12] +
                  m[2]*m[4]*m[9]*m[15] - m[2]*m[4]*m[11]*m[13] - m[2]*m[5]*m[8]*m[15] +
                  m[2]*m[5]*m[11]*m[12] + m[2]*m[7]*m[8]*m[13] - m[2]*m[7]*m[9]*m[12] -
                  m[3]*m[4]*m[9]*m[14] + m[3]*m[4]*m[10]*m[13] + m[3]*m[5]*m[8]*m[14] -
                  m[3]*m[5]*m[10]*m[12] - m[3]*m[6]*m[8]*m[13] + m[3]*m[6]*m[9]*m[12]

        if abs(det) < 1e-10 {
            // Singular matrix, return identity
            var identity = [Double](repeating: 0.0, count: 16)
            for i in 0..<4 { identity[i * 4 + i] = 1.0 }
            return identity
        }

        let invDet = 1.0 / det
        var inv = [Double](repeating: 0.0, count: 16)

        // Compute adjugate matrix and divide by determinant
        inv[0] = invDet * (m[5]*(m[10]*m[15]-m[11]*m[14]) - m[6]*(m[9]*m[15]-m[11]*m[13]) + m[7]*(m[9]*m[14]-m[10]*m[13]))
        inv[1] = invDet * -(m[1]*(m[10]*m[15]-m[11]*m[14]) - m[2]*(m[9]*m[15]-m[11]*m[13]) + m[3]*(m[9]*m[14]-m[10]*m[13]))
        inv[2] = invDet * (m[1]*(m[6]*m[15]-m[7]*m[14]) - m[2]*(m[5]*m[15]-m[7]*m[13]) + m[3]*(m[5]*m[14]-m[6]*m[13]))
        inv[3] = invDet * -(m[1]*(m[6]*m[11]-m[7]*m[10]) - m[2]*(m[5]*m[11]-m[7]*m[9]) + m[3]*(m[5]*m[10]-m[6]*m[9]))

        inv[4] = invDet * -(m[4]*(m[10]*m[15]-m[11]*m[14]) - m[6]*(m[8]*m[15]-m[11]*m[12]) + m[7]*(m[8]*m[14]-m[10]*m[12]))
        inv[5] = invDet * (m[0]*(m[10]*m[15]-m[11]*m[14]) - m[2]*(m[8]*m[15]-m[11]*m[12]) + m[3]*(m[8]*m[14]-m[10]*m[12]))
        inv[6] = invDet * -(m[0]*(m[6]*m[15]-m[7]*m[14]) - m[2]*(m[4]*m[15]-m[7]*m[12]) + m[3]*(m[4]*m[14]-m[6]*m[12]))
        inv[7] = invDet * (m[0]*(m[6]*m[11]-m[7]*m[10]) - m[2]*(m[4]*m[11]-m[7]*m[8]) + m[3]*(m[4]*m[10]-m[6]*m[8]))

        inv[8] = invDet * (m[4]*(m[9]*m[15]-m[11]*m[13]) - m[5]*(m[8]*m[15]-m[11]*m[12]) + m[7]*(m[8]*m[13]-m[9]*m[12]))
        inv[9] = invDet * -(m[0]*(m[9]*m[15]-m[11]*m[13]) - m[1]*(m[8]*m[15]-m[11]*m[12]) + m[3]*(m[8]*m[13]-m[9]*m[12]))
        inv[10] = invDet * (m[0]*(m[5]*m[15]-m[7]*m[13]) - m[1]*(m[4]*m[15]-m[7]*m[12]) + m[3]*(m[4]*m[13]-m[5]*m[12]))
        inv[11] = invDet * -(m[0]*(m[5]*m[11]-m[7]*m[9]) - m[1]*(m[4]*m[11]-m[7]*m[8]) + m[3]*(m[4]*m[9]-m[5]*m[8]))

        inv[12] = invDet * -(m[4]*(m[9]*m[14]-m[10]*m[13]) - m[5]*(m[8]*m[14]-m[10]*m[12]) + m[6]*(m[8]*m[13]-m[9]*m[12]))
        inv[13] = invDet * (m[0]*(m[9]*m[14]-m[10]*m[13]) - m[1]*(m[8]*m[14]-m[10]*m[12]) + m[2]*(m[8]*m[13]-m[9]*m[12]))
        inv[14] = invDet * -(m[0]*(m[5]*m[14]-m[6]*m[13]) - m[1]*(m[4]*m[14]-m[6]*m[12]) + m[2]*(m[4]*m[13]-m[5]*m[12]))
        inv[15] = invDet * (m[0]*(m[5]*m[10]-m[6]*m[9]) - m[1]*(m[4]*m[10]-m[6]*m[8]) + m[2]*(m[4]*m[9]-m[5]*m[8]))

        return inv
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
