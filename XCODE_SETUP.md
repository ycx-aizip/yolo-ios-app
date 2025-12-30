# Xcode Project Setup After Package Reorganization

## Changes Made
- Moved `Packages/` from `YOLOiOSApp/Packages/` to repo root `yolo-ios-app/Packages/`
- Moved `OpenCVBridge.h` and `OpenCVBridge.mm` to `Packages/OpenCV/`

## Required Xcode Changes

### 1. Update Build Settings in project.pbxproj

Open `YOLOiOSApp.xcodeproj` and update these build settings:

**FRAMEWORK_SEARCH_PATHS** (both Debug and Release):
```
Old: $(PROJECT_DIR)/Packages/OpenCV/ios
New: $(SRCROOT)/../Packages/OpenCV/ios
```

**For simulator builds:**
```
Old: $(PROJECT_DIR)/Packages/OpenCV/ios-simulator
New: $(SRCROOT)/../Packages/OpenCV/ios-simulator
```

**HEADER_SEARCH_PATHS** (if set):
```
Old: $(PROJECT_DIR)/Packages/OpenCV/ios/Headers
New: $(SRCROOT)/../Packages/OpenCV/ios/Headers
```

### 2. Re-add OpenCVBridge Files to Xcode Project

Since we moved `OpenCVBridge.h` and `OpenCVBridge.mm`:

1. **Remove old references** (if they show as red/missing in Xcode):
   - Select the files in Xcode navigator
   - Press Delete → "Remove Reference"

2. **Add files from new location**:
   - Right-click on `YOLOiOSApp` group
   - "Add Files to YOLOiOSApp..."
   - Navigate to `../Packages/OpenCV/`
   - Select `OpenCVBridge.h` and `OpenCVBridge.mm`
   - ✅ **IMPORTANT**: Uncheck "Copy items if needed"
   - ✅ Ensure "Add to targets: YOLOiOSApp" is checked
   - Click "Add"

### 3. Verify Bridging Header

The bridging header should already import OpenCVBridge:
```objc
#import "OpenCVBridge.h"
```

This will work because Xcode adds the parent directory of source files to the header search path.

### 4. Clean and Rebuild

```bash
# Clean build folder
Product → Clean Build Folder (Cmd+Shift+K)

# Rebuild
Product → Build (Cmd+B)
```

## Verification

The build should succeed with these changes. Verify:
- No "file not found" errors for OpenCVBridge.h
- OpenCV framework links correctly
- App runs in simulator

## For Future Projects

When using AizipFishCount in a new project:
1. Copy `Packages/OpenCV/` to your project
2. Add `OpenCVBridge.h/.mm` to your Xcode target (as above)
3. Configure framework search paths to point to `Packages/OpenCV/`
4. Import in bridging header
