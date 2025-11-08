# AWS EC2 Deployment Guide with GPU Support

This guide will walk you through deploying your FastAPI backend to AWS EC2 with GPU support for ML model inference.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [AWS EC2 Setup](#aws-ec2-setup)
3. [Instance Configuration](#instance-configuration)
4. [Application Deployment](#application-deployment)
5. [GPU Configuration](#gpu-configuration)
6. [Adding ML Models](#adding-ml-models)
7. [Production Best Practices](#production-best-practices)

---

## Prerequisites

### Local Setup
- [ ] AWS Account with billing enabled
- [ ] AWS CLI installed (`aws --version`)
- [ ] SSH key pair for EC2 access
- [ ] Git repository (GitHub/GitLab)
- [ ] Basic understanding of Linux commands

### AWS Services Needed
- **EC2** - For compute instances
- **Security Groups** - For firewall rules
- **Elastic IP** (optional) - For static IP address
- **IAM** - For permissions
- **CloudWatch** (optional) - For monitoring

---

## AWS EC2 Setup

### Step 1: Choose GPU Instance Type

For ML workloads with GPU, consider these instance types:

| Instance Type | GPU | vCPU | Memory | Cost (approx/hour) | Best For |
|--------------|-----|------|--------|-------------------|----------|
| **g4dn.xlarge** | 1x NVIDIA T4 (16GB) | 4 | 16 GB | $0.526 | Development/Testing |
| **g4dn.2xlarge** | 1x NVIDIA T4 (16GB) | 8 | 32 GB | $0.752 | Small production |
| **g5.xlarge** | 1x NVIDIA A10G (24GB) | 4 | 16 GB | $1.006 | Modern ML models |
| **p3.2xlarge** | 1x NVIDIA V100 (16GB) | 8 | 61 GB | $3.06 | Training/Large models |

**Recommendation for starting**: `g4dn.xlarge` - Good balance of cost and performance

### Step 2: Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Configure Instance**:
   - **Name**: `composer-pro-backend-gpu`
   - **AMI**: Select **Deep Learning AMI GPU PyTorch** (Ubuntu)
     - Search for: "Deep Learning AMI GPU PyTorch 2.0" in AWS Marketplace
     - This comes pre-installed with NVIDIA drivers, CUDA, and cuDNN
   
3. **Instance Type**: Select `g4dn.xlarge` (or your choice)

4. **Key Pair**: Create or select existing SSH key pair
   - Download `.pem` file and save securely
   - Set permissions: `chmod 400 your-key.pem`

5. **Network Settings**:
   - Create security group with these rules:
     - **SSH** (22) - Your IP only
     - **HTTP** (80) - 0.0.0.0/0
     - **HTTPS** (443) - 0.0.0.0/0
     - **Custom TCP** (8000) - 0.0.0.0/0 (for API)

6. **Storage**: 
   - Increase to **50-100 GB** (ML models can be large)
   - Choose **gp3** for better performance

7. **Advanced Details**:
   - Enable detailed monitoring (optional)

8. Click **Launch Instance**

### Step 3: Connect to Instance

```bash
# Save your key with correct permissions
chmod 400 your-key.pem

# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip

# Example:
# ssh -i ~/.ssh/composer-key.pem ubuntu@54.123.45.67
```

---

## Instance Configuration

### Step 1: Update System

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y git htop tmux nginx certbot python3-certbot-nginx
```

### Step 2: Verify GPU Setup

```bash
# Check NVIDIA driver
nvidia-smi

# Should show something like:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
# | N/A   32C    P0    15W /  70W |      0MiB / 15360MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# Check CUDA version
nvcc --version
```

### Step 3: Setup Application User (Optional but Recommended)

```bash
# Create application user
sudo useradd -m -s /bin/bash appuser
sudo usermod -aG docker appuser  # If using Docker
```

---

## Application Deployment

### Option A: Direct Deployment (Recommended for GPU)

#### 1. Clone Repository

```bash
cd /home/ubuntu
git clone git@github.com:yiiiq/composer-pro-backend.git
cd composer-pro-backend
```

#### 2. Create Virtual Environment

```bash
# Create venv
python3 -m venv .venv --copies

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

#### 4. Configure Environment

```bash
# Create .env file
cp .env.example .env

# Edit with your settings
nano .env
```

Set these values in `.env`:
```bash
APP_NAME=Compose Pro Backend
DEBUG=False
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

#### 5. Test Application

```bash
# Run in foreground to test
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test from another terminal or browser
# http://your-ec2-ip:8000/docs
```

#### 6. Setup Systemd Service (Production)

Create service file:
```bash
sudo nano /etc/systemd/system/composer-backend.service
```

Add this content:
```ini
[Unit]
Description=Composer Pro Backend API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/composer-pro-backend
Environment="PATH=/home/ubuntu/composer-pro-backend/.venv/bin"
ExecStart=/home/ubuntu/composer-pro-backend/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start service:
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable composer-backend

# Start service
sudo systemctl start composer-backend

# Check status
sudo systemctl status composer-backend

# View logs
sudo journalctl -u composer-backend -f
```

### Option B: Docker Deployment

⚠️ **Note**: For GPU support, you need nvidia-docker runtime

#### 1. Install Docker with NVIDIA Support

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Update Dockerfile for GPU

The Dockerfile needs CUDA base image - I'll create one below.

#### 3. Build and Run

```bash
# Build image
docker build -t composer-backend:gpu -f Dockerfile.gpu .

# Run with GPU support
docker run -d \
  --name composer-backend \
  --gpus all \
  -p 8000:8000 \
  --restart unless-stopped \
  composer-backend:gpu

# Check logs
docker logs -f composer-backend

# Test GPU inside container
docker exec -it composer-backend nvidia-smi
```

---

## GPU Configuration

### Verify GPU is Accessible in Python

Create test script:
```bash
nano test_gpu.py
```

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

Run it:
```bash
source .venv/bin/activate
python test_gpu.py
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: Tesla T4
GPU memory: 15.36 GB
```

---

## Adding ML Models

### Step 1: Prepare Your Model

Save your trained model:
```python
import torch

# Example: Save PyTorch model
model = YourModel()
torch.save(model.state_dict(), 'model.pth')

# Or save entire model
torch.save(model, 'model_full.pth')
```

### Step 2: Upload Model to EC2

```bash
# Option 1: SCP from local machine
scp -i your-key.pem model.pth ubuntu@your-ec2-ip:/home/ubuntu/composer-pro-backend/app/models/saved_models/

# Option 2: Download from S3
aws s3 cp s3://your-bucket/model.pth /home/ubuntu/composer-pro-backend/app/models/saved_models/

# Option 3: Git LFS (for models in repo)
cd /home/ubuntu/composer-pro-backend
git lfs pull
```

### Step 3: Update Model Loader

Edit `app/models/model_loader.py` to load your model with GPU support:

```python
import torch
from pathlib import Path

class ModelLoader:
    def __init__(self, model_dir: str = "./app/models/saved_models"):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
    
    def load_model(self, model_name: str):
        """Load model to GPU if available"""
        model_path = self.model_dir / f"{model_name}.pth"
        
        if model_name in self.models:
            return self.models[model_name]
        
        # Load your model
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        self.models[model_name] = model
        return model
    
    def predict(self, model_name: str, input_data):
        """Run inference on GPU"""
        model = self.load_model(model_name)
        
        with torch.no_grad():
            # Move input to GPU
            input_tensor = torch.tensor(input_data).to(self.device)
            
            # Run prediction
            output = model(input_tensor)
            
            # Move back to CPU for response
            return output.cpu().numpy()
```

### Step 4: Test Model Inference

```bash
# Restart service
sudo systemctl restart composer-backend

# Test API
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [1, 2, 3], "model_name": "your_model"}'
```

---

## Production Best Practices

### 1. Setup Nginx Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/composer-backend
```

Add:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # or use IP
    
    client_max_body_size 100M;  # For large model uploads
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout for long-running predictions
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/composer-backend /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 2. Setup SSL with Let's Encrypt

```bash
# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
sudo certbot renew --dry-run
```

### 3. Setup Monitoring

```bash
# Install htop for system monitoring
htop

# Monitor GPU
watch -n 1 nvidia-smi

# Monitor logs
sudo journalctl -u composer-backend -f
```

### 4. Setup Auto-scaling (Optional)

For production, consider:
- **AWS Auto Scaling Groups** for multiple instances
- **Application Load Balancer** for traffic distribution
- **CloudWatch Alarms** for scaling triggers
- **Elastic Container Service (ECS)** for Docker orchestration

### 5. Cost Optimization

```bash
# Stop instance when not in use (saves GPU costs)
aws ec2 stop-instances --instance-ids i-1234567890abcdef0

# Start when needed
aws ec2 start-instances --instance-ids i-1234567890abcdef0

# Or use AWS Instance Scheduler
# Or use EC2 Spot Instances for 70% cost savings
```

### 6. Backup Strategy

```bash
# Automated backups with cron
crontab -e

# Add: Backup models daily at 2 AM
0 2 * * * tar -czf /home/ubuntu/backups/models-$(date +\%Y\%m\%d).tar.gz /home/ubuntu/composer-pro-backend/app/models/saved_models/
```

### 7. Security Hardening

```bash
# Update security group to restrict SSH to your IP only
# Use AWS Systems Manager Session Manager instead of SSH
# Enable CloudWatch Logs
# Setup AWS WAF for API protection
# Use AWS Secrets Manager for sensitive data
```

---

## Quick Deployment Checklist

- [ ] Launch EC2 GPU instance (g4dn.xlarge)
- [ ] Configure security groups (ports 22, 80, 443, 8000)
- [ ] SSH into instance
- [ ] Verify GPU with `nvidia-smi`
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies with GPU support
- [ ] Upload ML models
- [ ] Configure environment variables
- [ ] Test application locally
- [ ] Setup systemd service
- [ ] Configure Nginx reverse proxy
- [ ] Setup SSL certificate
- [ ] Test API endpoints
- [ ] Monitor GPU usage and logs
- [ ] Setup backups and monitoring

---

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall if needed
sudo apt purge nvidia-*
sudo apt install nvidia-driver-525

# Reboot
sudo reboot
```

### Out of GPU Memory
```python
# Clear GPU cache in your code
import torch
torch.cuda.empty_cache()

# Use smaller batch sizes
# Enable gradient checkpointing
# Use model quantization
```

### Service Won't Start
```bash
# Check logs
sudo journalctl -u composer-backend -n 50

# Check if port is in use
sudo lsof -i :8000

# Test manually
cd /home/ubuntu/composer-pro-backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Next Steps

1. **Deploy your first model** following the guide above
2. **Setup monitoring** with CloudWatch
3. **Implement CI/CD** with GitHub Actions
4. **Add authentication** to your API
5. **Setup database** for storing predictions
6. **Implement caching** with Redis
7. **Add model versioning**

---

## Estimated Costs

For `g4dn.xlarge` instance:
- **On-Demand**: ~$380/month (24/7)
- **Reserved (1-year)**: ~$250/month (35% savings)
- **Spot Instance**: ~$115/month (70% savings, can be interrupted)

💡 **Tip**: Use Spot Instances for development/testing and Reserved for production.

---

## Support Resources

- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)
- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch GPU Documentation](https://pytorch.org/docs/stable/cuda.html)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
