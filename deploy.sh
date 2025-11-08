#!/bin/bash

###############################################################################
# AWS EC2 Deployment Script for Composer Pro Backend
# This script automates the deployment process on EC2
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/home/ubuntu/composer-pro-backend"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="composer-backend"

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Composer Pro Backend Deployment${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if running as correct user
if [ "$USER" != "ubuntu" ]; then
    echo -e "${YELLOW}Warning: This script is designed to run as 'ubuntu' user${NC}"
fi

# Update system
echo -e "\n${GREEN}[1/8] Updating system packages...${NC}"
sudo apt update && sudo apt upgrade -y

# Install required packages
echo -e "\n${GREEN}[2/8] Installing required packages...${NC}"
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx

# Check NVIDIA GPU
echo -e "\n${GREEN}[3/8] Checking GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: No GPU detected. Will run on CPU.${NC}"
fi

# Clone or update repository
echo -e "\n${GREEN}[4/8] Setting up application...${NC}"
if [ -d "$APP_DIR" ]; then
    echo "Repository exists. Pulling latest changes..."
    cd $APP_DIR
    git pull
else
    echo "Cloning repository..."
    cd /home/ubuntu
    git clone git@github.com:yiiiq/composer-pro-backend.git
    cd $APP_DIR
fi

# Create virtual environment
echo -e "\n${GREEN}[5/8] Setting up Python virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment exists. Skipping creation."
else
    python3.11 -m venv $VENV_DIR --copies
fi

# Activate virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo -e "\n${GREEN}[6/8] Installing Python dependencies...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

pip install -r requirements.txt

# Setup environment file
echo -e "\n${GREEN}[7/8] Configuring environment...${NC}"
if [ ! -f "$APP_DIR/.env" ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env file with your configuration${NC}"
else
    echo ".env file already exists. Skipping."
fi

# Setup systemd service
echo -e "\n${GREEN}[8/8] Setting up systemd service...${NC}"
sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Composer Pro Backend API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
Environment="PATH=$VENV_DIR/bin"
ExecStart=$VENV_DIR/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable $SERVICE_NAME

# Start or restart service
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "Restarting service..."
    sudo systemctl restart $SERVICE_NAME
else
    echo "Starting service..."
    sudo systemctl start $SERVICE_NAME
fi

# Check service status
sleep 2
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✓ Service is running${NC}"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    sudo systemctl status $SERVICE_NAME
    exit 1
fi

# Test API
echo -e "\n${GREEN}Testing API...${NC}"
sleep 3
if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ API is responding${NC}"
else
    echo -e "${YELLOW}⚠ API test failed. Check logs with: sudo journalctl -u $SERVICE_NAME -f${NC}"
fi

echo -e "\n${GREEN}======================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "\nUseful commands:"
echo -e "  View logs:     ${YELLOW}sudo journalctl -u $SERVICE_NAME -f${NC}"
echo -e "  Service status: ${YELLOW}sudo systemctl status $SERVICE_NAME${NC}"
echo -e "  Restart:       ${YELLOW}sudo systemctl restart $SERVICE_NAME${NC}"
echo -e "  Stop:          ${YELLOW}sudo systemctl stop $SERVICE_NAME${NC}"
echo -e "\nAPI Documentation: ${YELLOW}http://$(curl -s ifconfig.me):8000/docs${NC}"
echo -e "\n${YELLOW}Don't forget to:${NC}"
echo -e "  1. Edit .env file with your configuration"
echo -e "  2. Setup Nginx reverse proxy"
echo -e "  3. Configure SSL certificate"
echo -e "  4. Upload your ML models to $APP_DIR/app/models/saved_models/"
