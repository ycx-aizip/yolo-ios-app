#!/bin/bash
#
# Build AizipFishCount.xcframework
#
# This script builds the AizipFishCount backend framework for both device and simulator
# Output: AizipFishCount.xcframework (universal framework)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Building AizipFishCount.xcframework${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

# Configuration
SCHEME_NAME="AizipFishCount"
PROJECT_PATH="$REPO_ROOT/Package.swift"
BUILD_DIR="$REPO_ROOT/build"
ARCHIVES_DIR="$BUILD_DIR/archives"
OUTPUT_DIR="$REPO_ROOT/releases"
FRAMEWORK_NAME="AizipFishCount"

# Derived paths
IOS_ARCHIVE="$ARCHIVES_DIR/ios.xcarchive"
SIMULATOR_ARCHIVE="$ARCHIVES_DIR/ios-simulator.xcarchive"
XCFRAMEWORK_PATH="$OUTPUT_DIR/$FRAMEWORK_NAME.xcframework"

# Clean build directories
echo -e "${YELLOW}Cleaning build directories...${NC}"
rm -rf "$BUILD_DIR"
rm -rf "$OUTPUT_DIR"
mkdir -p "$ARCHIVES_DIR"
mkdir -p "$OUTPUT_DIR"

# Build for iOS Device (arm64)
echo -e "\n${YELLOW}Building for iOS Device (arm64)...${NC}"
xcodebuild archive \
    -scheme "$SCHEME_NAME" \
    -destination "generic/platform=iOS" \
    -archivePath "$IOS_ARCHIVE" \
    -derivedDataPath "$BUILD_DIR/DerivedData" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    OTHER_SWIFT_FLAGS="-no-verify-emitted-module-interface"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to build for iOS Device${NC}"
    exit 1
fi

# Build for iOS Simulator (arm64 + x86_64)
echo -e "\n${YELLOW}Building for iOS Simulator (arm64, x86_64)...${NC}"
xcodebuild archive \
    -scheme "$SCHEME_NAME" \
    -destination "generic/platform=iOS Simulator" \
    -archivePath "$SIMULATOR_ARCHIVE" \
    -derivedDataPath "$BUILD_DIR/DerivedData" \
    SKIP_INSTALL=NO \
    BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
    OTHER_SWIFT_FLAGS="-no-verify-emitted-module-interface"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to build for iOS Simulator${NC}"
    exit 1
fi

# Create XCFramework
echo -e "\n${YELLOW}Creating XCFramework...${NC}"

# Remove old xcframework if exists
if [ -d "$XCFRAMEWORK_PATH" ]; then
    rm -rf "$XCFRAMEWORK_PATH"
fi

xcodebuild -create-xcframework \
    -framework "$IOS_ARCHIVE/Products/Library/Frameworks/$FRAMEWORK_NAME.framework" \
    -framework "$SIMULATOR_ARCHIVE/Products/Library/Frameworks/$FRAMEWORK_NAME.framework" \
    -output "$XCFRAMEWORK_PATH"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Successfully created AizipFishCount.xcframework${NC}"
    echo -e "${GREEN}Output: $XCFRAMEWORK_PATH${NC}\n"

    # Show framework size
    FRAMEWORK_SIZE=$(du -sh "$XCFRAMEWORK_PATH" | cut -f1)
    echo -e "${BLUE}Framework size: $FRAMEWORK_SIZE${NC}"

    # Show contents
    echo -e "\n${YELLOW}XCFramework structure:${NC}"
    tree -L 2 "$XCFRAMEWORK_PATH" 2>/dev/null || ls -lR "$XCFRAMEWORK_PATH"

    echo -e "\n${GREEN}Build complete!${NC}"
    echo -e "${GREEN}Framework ready at: $XCFRAMEWORK_PATH${NC}"
else
    echo -e "${RED}❌ Failed to create xcframework${NC}"
    exit 1
fi

# Clean up archives (optional - save space)
# Uncomment to delete archives after successful build
# rm -rf "$ARCHIVES_DIR"
