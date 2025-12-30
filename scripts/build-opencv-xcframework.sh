#!/bin/bash
#
# Build opencv2.xcframework from separate architecture frameworks
#
# This script creates a universal xcframework from the ios and ios-simulator frameworks
# Resolves "Embed & Sign" code signing issues
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building opencv2.xcframework...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$SCRIPT_DIR/.."

# Paths
OPENCV_DIR="$REPO_ROOT/Packages/opencv2"
IOS_FRAMEWORK="$OPENCV_DIR/ios/opencv2.framework"
SIMULATOR_FRAMEWORK="$OPENCV_DIR/ios-simulator/opencv2.framework"
OUTPUT_XCFRAMEWORK="$OPENCV_DIR/opencv2.xcframework"

# Verify source frameworks exist
if [ ! -d "$IOS_FRAMEWORK" ]; then
    echo -e "${RED}Error: iOS framework not found at $IOS_FRAMEWORK${NC}"
    exit 1
fi

if [ ! -d "$SIMULATOR_FRAMEWORK" ]; then
    echo -e "${RED}Error: Simulator framework not found at $SIMULATOR_FRAMEWORK${NC}"
    exit 1
fi

echo -e "${YELLOW}Source frameworks:${NC}"
echo "  iOS: $IOS_FRAMEWORK"
echo "  Simulator: $SIMULATOR_FRAMEWORK"

# Clean up old xcframework if it exists
if [ -d "$OUTPUT_XCFRAMEWORK" ]; then
    echo -e "${YELLOW}Removing old xcframework...${NC}"
    rm -rf "$OUTPUT_XCFRAMEWORK"
fi

# Create the xcframework
echo -e "${YELLOW}Creating xcframework...${NC}"
xcodebuild -create-xcframework \
    -framework "$IOS_FRAMEWORK" \
    -framework "$SIMULATOR_FRAMEWORK" \
    -output "$OUTPUT_XCFRAMEWORK"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Successfully created opencv2.xcframework${NC}"
    echo -e "${GREEN}Output: $OUTPUT_XCFRAMEWORK${NC}"

    # Show framework info
    echo -e "\n${YELLOW}Framework contents:${NC}"
    ls -lh "$OUTPUT_XCFRAMEWORK"

    echo -e "\n${GREEN}You can now:${NC}"
    echo "  1. Remove opencv2.framework from Xcode project"
    echo "  2. Add opencv2.xcframework instead"
    echo "  3. Set to 'Embed & Sign'"
else
    echo -e "${RED}❌ Failed to create xcframework${NC}"
    exit 1
fi
