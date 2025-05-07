# Fish Counting Tracking System

## Overview

The tracking system in this application uses an optimized implementation of ByteTrack for robust fish tracking across video frames. The system supports multiple camera viewing directions (top-to-bottom, bottom-to-top, left-to-right, right-to-left) and is designed to maintain consistent tracking through occlusions, temporary disappearances, and variable movement patterns.

## Key Components

- **ByteTracker**: Core tracking algorithm that maintains track IDs across frames
- **STrack**: Single track representation with Kalman filtering for motion prediction
- **TrackingDetector**: Integrates tracking with threshold crossing detection
- **MatchingUtils**: Provides utility methods for detection-track association
- **KalmanFilter**: State estimation for predicting fish positions between detections
- **TrackingUtils**: Centralized parameter management for the tracking system

## Tracking Algorithm Flow

1. **Object Detection**: YOLO-based detection provides bounding boxes for each fish
2. **Detection Association**: ByteTrack algorithm associates detections across frames using both IoU and position matching
3. **Track Management**: Maintains track lifecycle (new → tracked → lost → removed)
4. **Threshold Crossing**: Detects when fish cross defined threshold lines for counting
5. **Direction-aware Processing**: Adapts to different camera viewing angles

## Track Lifecycle

- **Potential**: Detections that persist for multiple frames become candidate tracks
- **New**: Newly created tracks that haven't been confirmed yet
- **Tracked**: Active tracks currently associated with detections
- **Lost**: Tracks not matched in recent frames, may be recovered
- **Removed**: Tracks that have been lost for too long and are no longer considered

## TTL (Time-To-Live) System

Tracks use an adaptive Time-To-Live (TTL) system to determine how long they should persist when not matched with new detections:

- Each track has a TTL counter that is decremented when not matched with a detection
- TTL values are adaptive based on track consistency and movement patterns
- Higher TTL values (15-20) for fish with consistent expected movement patterns
- Lower TTL values (8-12) for erratic movement or tracks being reactivated
- When TTL reaches zero, tracks transition from tracked to lost state
- Lost tracks are removed entirely after `maxTimeLost` frames

## Configurable Parameters

All tracking parameters are centralized in `TrackingUtils.swift` for easy configuration and tuning.

### Association Thresholds

| Parameter | Description | Default |
|-----------|-------------|---------|
| `highMatchThreshold` | IoU threshold for first-stage matching | 0.3 |
| `lowMatchThreshold` | IoU threshold for second-stage matching | 0.25 |
| `minMatchDistance` | Minimum distance for track association | 0.3 |
| `iouMatchThreshold` | IoU threshold for immediate association | 0.4 |

### Track Lifecycle

| Parameter | Description | Default |
|-----------|-------------|---------|
| `maxTimeLost` | Maximum frames to keep lost tracks | 15 |
| `maxUnmatchedFrames` | Maximum frames to keep potential tracks | 15 |
| `requiredFramesForTrack` | Frames required for track creation | 1 |
| `maxMatchingDistance` | Maximum distance for matching potential tracks | 0.6 |

### TTL Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `defaultTTL` | Default TTL for new tracks | 15 |
| `highConsistencyTTL` | TTL for tracks with consistent movement | 20 |
| `mediumConsistencyTTL` | TTL for tracks with moderate consistency | 15 |
| `lowConsistencyTTL` | TTL for tracks with erratic movement | 10 |
| `reactivationHighTTL` | TTL for reactivated tracks (high consistency) | 15 |
| `reactivationMediumTTL` | TTL for reactivated tracks (medium consistency) | 12 |
| `reactivationLowTTL` | TTL for reactivated tracks (low consistency) | 8 |

### Movement Constraints

| Parameter | Description | Default |
|-----------|-------------|---------|
| `maxHorizontalDeviation` | Maximum allowed horizontal deviation | 0.2 |
| `maxVerticalDeviation` | Maximum allowed vertical deviation | 0.2 |
| `consistencyIncreaseRate` | Rate to increase movement consistency | 0.2 |
| `consistencyDecreaseRate` | Rate to decrease consistency | 0.1 |
| `reactivationConsistencyIncreaseRate` | Consistency increase rate for reactivation | 0.15 |
| `reactivationConsistencyDecreaseRate` | Consistency decrease rate for reactivation | 0.2 |

## Direction-Specific Parameters

The tracking system supports different parameter sets for each counting direction:

- `topToBottom`: Default fish movement from top to bottom of frame
- `bottomToTop`: Fish movement from bottom to top of frame
- `leftToRight`: Fish movement from left to right of frame
- `rightToLeft`: Fish movement from right to left of frame

Parameters can be tuned for each direction by modifying the appropriate thresholds in the `directionThresholds` dictionary in `TrackingUtils.swift`.

## Tuning Guidelines

When tuning the tracking system for specific environments:

1. Start with the **Association Thresholds** to ensure proper detection-track matching
2. Adjust **TTL Parameters** based on fish movement patterns and density
3. Fine-tune **Movement Constraints** to match expected movement directions
4. Consider direction-specific adjustments for each camera angle

For crowded scenes (many fish), consider:
- Lower TTL values to prevent ID switches
- Higher association thresholds for more confident matching
- Stricter movement consistency requirements

For sparse scenes (few fish):
- Higher TTL values to maintain tracks through longer occlusions
- Lower thresholds to recover tracks more aggressively 