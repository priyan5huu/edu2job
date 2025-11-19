#!/bin/bash

# Edu2Job - Stop Script
# This script stops the backend server

echo "🛑 Stopping Edu2Job..."
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if PID file exists
if [ -f ".server_pid" ]; then
    SERVER_PID=$(cat .server_pid)
    
    # Check if process is running
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping server (PID: $SERVER_PID)...${NC}"
        kill $SERVER_PID 2>/dev/null
        sleep 1
        
        # Force kill if still running
        if ps -p $SERVER_PID > /dev/null 2>&1; then
            kill -9 $SERVER_PID 2>/dev/null
        fi
        
        echo -e "${GREEN}✓ Server stopped${NC}"
    else
        echo -e "${YELLOW}⚠️  Server process not found (PID: $SERVER_PID)${NC}"
    fi
    
    # Remove PID file
    rm -f .server_pid
else
    echo -e "${YELLOW}⚠️  No PID file found${NC}"
fi

# Kill any process on port 8000 (cleanup)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}Cleaning up port 8000...${NC}"
    lsof -ti :8000 | xargs kill -9 2>/dev/null
    echo -e "${GREEN}✓ Port 8000 cleared${NC}"
fi

echo ""
echo -e "${GREEN}Edu2Job server stopped successfully! 👋${NC}"
echo ""
