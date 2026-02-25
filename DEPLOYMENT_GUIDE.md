# 🚀 Deployment Guide — Maternity Readmission Prediction System

Complete instructions for deploying the Streamlit dashboard in all environments.

---

## Option 1: Streamlit Cloud (Recommended — Free, 3 Minutes)

### Step-by-Step

1. **Create a GitHub repository**
   ```
   New repo → e.g. "maternity-readmission"
   ```

2. **Upload all project files** to the repo root:
   ```
   maternity_data.csv
   streamlit_app.py
   requirements.txt
   readmission_model.ipynb
   ethics_audit_report.pdf
   README.md
   DEPLOYMENT_GUIDE.md
   ```

3. **Deploy on Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Click **"New App"**
   - Select your GitHub repo
   - Set **Main file path:** `streamlit_app.py`
   - Click **Deploy**

4. **App goes live** at: `https://[your-username]-[repo-name]-streamlit.app`

### Advantages
- Free hosting (community tier)
- Auto-redeploys on GitHub push
- No server management required
- HTTPS included

---

## Option 2: Local Development (For Testing)

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone or download project files
cd maternity-readmission

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py

# Access at: http://localhost:8501
```

### Development Mode

```bash
# Run with auto-reload on file changes
streamlit run streamlit_app.py --server.runOnSave true
```

---

## Option 3: Docker (Production)

### Dockerfile

Create a file named `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
```

### Build & Run

```bash
# Build image
docker build -t maternity-readmission .

# Run container
docker run -p 8501:8501 maternity-readmission

# Run in background
docker run -d -p 8501:8501 --name maternity maternity-readmission

# Access at: http://localhost:8501
```

### Docker Compose (with volume mounting)

```yaml
version: '3.8'
services:
  maternity-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./maternity_data.csv:/app/maternity_data.csv:ro
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_MAX_UPLOAD_SIZE=50
```

```bash
docker-compose up -d
```

---

## Option 4: Cloud Platforms

### AWS EC2 / Google Cloud / Azure VM

```bash
# On your cloud VM (Ubuntu 22.04 example)
sudo apt-get update
sudo apt-get install -y python3 python3-pip

# Upload files via scp or git clone
pip3 install -r requirements.txt

# Run with nohup (persist after terminal close)
nohup streamlit run streamlit_app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 > app.log 2>&1 &

# Or use screen
screen -S maternity
streamlit run streamlit_app.py --server.port 8501
# Ctrl+A, D to detach
```

---

## 🔒 Security Considerations

### For Production Healthcare Deployment

| Requirement | Implementation |
|-------------|---------------|
| HTTPS | Use reverse proxy (nginx + Let's Encrypt) or platform HTTPS |
| Authentication | Add Streamlit auth or OAuth2 (Keycloak, Azure AD) |
| Data encryption (at rest) | AES-256 for CSV files and databases |
| Data encryption (in transit) | TLS 1.3 minimum |
| Audit logging | Log all predictions with timestamp and user |
| Access control | Role-based access — clinicians only |
| Data retention | Define and enforce data retention policy |
| HIPAA/GDPR | Consult legal team before handling real patient data |

### Nginx Reverse Proxy (HTTPS)

```nginx
server {
    listen 443 ssl;
    server_name your-hospital-domain.com;

    ssl_certificate     /etc/letsencrypt/live/your-domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

---

## 📊 Monitoring Setup

### Application Health

```bash
# Check if app is running
curl http://localhost:8501/_stcore/health

# View live logs (Docker)
docker logs -f maternity

# View live logs (nohup)
tail -f app.log
```

### Model Performance Monitoring (Recommended Script)

```python
# monitor.py — run quarterly
import pandas as pd
from sklearn.metrics import accuracy_score

# Load new data + model predictions
# Check subgroup accuracy gaps
# Alert if gap > 10% or overall accuracy drops > 3%
```

---

## 🧪 Testing Before Deployment

```bash
# 1. Verify all files present
ls -la maternity_data.csv streamlit_app.py requirements.txt

# 2. Install and test locally first
pip install -r requirements.txt
streamlit run streamlit_app.py

# 3. Test predictions manually
#    - Enter low-risk patient → expect < 40% probability
#    - Enter high-risk patient (complications, long stay, rural) → expect > 60%

# 4. Verify bias audit tab loads correctly

# 5. Check model loads without errors in console
```

---

## ✅ Production Checklist

```
Pre-deployment:
  [ ] IRB/Ethics Committee approval obtained
  [ ] Informed consent protocol documented
  [ ] Data security audit completed
  [ ] AES-256 encryption implemented
  [ ] HTTPS configured
  [ ] Authentication added
  [ ] Local hospital data validation completed
  [ ] Clinical team validation sign-off

Deployment:
  [ ] Docker container health check passing
  [ ] HTTPS working
  [ ] All 5 dashboard tabs functional
  [ ] Predictions consistent with expected values
  [ ] Audit logging enabled

Post-deployment:
  [ ] Clinical staff training completed
  [ ] Monitoring alerts configured
  [ ] Adverse event reporting workflow live
  [ ] Quarterly bias audit scheduled
  [ ] Incident response plan documented
```

---

## 📞 Troubleshooting

| Error | Solution |
|-------|----------|
| `maternity_data.csv not found` | Ensure CSV is in same directory as `streamlit_app.py` |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Port 8501 already in use` | Run `streamlit run streamlit_app.py --server.port 8502` |
| Slow model loading | First load trains model (~5 sec); subsequent loads use cache |
| Docker build fails | Check Python version (requires 3.9+) |

---

**Version:** 1.0 | **Date:** February 2024
