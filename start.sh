#!/bin/bash

# Edu2Job - Quick Start Script
# This script starts the backend server and opens the frontend in your browser

echo "🚀 Starting Edu2Job..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found!${NC}"
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
    echo ""
fi

# Activate virtual environment
echo -e "${BLUE}📦 Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if dependencies are installed
if [ ! -f ".venv/lib/python*/site-packages/flask/__init__.py" ]; then
    echo -e "${YELLOW}📥 Installing dependencies...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
    echo ""
fi

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${YELLOW}⚠️  Port 8000 is already in use${NC}"
    echo "Stopping existing process..."
    lsof -ti :8000 | xargs kill -9 2>/dev/null
    sleep 1
    echo -e "${GREEN}✓ Port 8000 cleared${NC}"
    echo ""
fi

# Start the backend server in the background
echo -e "${BLUE}🔧 Starting Flask backend server...${NC}"
cd backend

# Start server and capture PID
python app.py > ../logs/server_output.log 2>&1 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Check if server is running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Backend server started successfully${NC}"
    echo -e "${GREEN}  Server PID: $SERVER_PID${NC}"
    echo -e "${GREEN}  Running on: http://localhost:8000${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  Server failed to start. Check logs/server_output.log for details${NC}"
    exit 1
fi

# Go back to root directory
cd ..

# Save PID to file for easy stopping later
echo $SERVER_PID > .server_pid

# Open frontend in default browser
echo -e "${BLUE}🌐 Opening frontend in browser...${NC}"
sleep 1

# Detect OS and open browser accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "frontend/login.html"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "frontend/login.html" 2>/dev/null
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    # Windows
    start "frontend/login.html"
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ Edu2Job is now running!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}📍 Backend API:${NC}  http://localhost:8000"
echo -e "${BLUE}📍 Frontend:${NC}     frontend/login.html (opened in browser)"
echo ""
echo -e "${YELLOW}To stop the server:${NC}"
echo -e "  ./stop.sh"
echo -e "  or press Ctrl+C in this terminal"
echo ""
echo -e "${BLUE}Server logs:${NC} logs/server_output.log"
echo ""
echo -e "${GREEN}Happy predicting! 🎯${NC}"
echo ""

# Keep script running to show logs (optional)
echo -e "${BLUE}Press Ctrl+C to stop the server...${NC}"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping server...'; kill $SERVER_PID 2>/dev/null; rm -f .server_pid; echo 'Server stopped.'; exit 0" INT TERM

# Keep the script running and show live logs
tail -f logs/server_output.log 2>/dev/null &
wait $SERVER_PID
