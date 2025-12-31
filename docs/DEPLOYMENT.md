# Deployment Guide

## Pre-Deployment Checklist

Before pushing to production:

- [ ] Models trained and validated
- [ ] Performance meets minimum requirements (MAPE <50%)
- [ ] Streamlit app tested locally
- [ ] Documentation reviewed
- [ ] .gitignore configured
- [ ] Sensitive data excluded
- [ ] Dependencies documented

---

## Local Development

### Setup

```bash
cd ag_iq_ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Mac only
brew install libomp
```

### Running Locally

```bash
streamlit run app.py
# Access at http://localhost:8501
```

---

## Streamlit Cloud Deployment (Recommended)

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- Repository pushed to GitHub

### Steps

1. **Prepare Repository**

```bash
# Ensure .gitignore excludes large files
# Models and data should NOT be in git (too large)
git add .
git commit -m "Ag IQ valuation system"
git push origin main
```

2. **Upload Models Separately**

Models are too large for GitHub. Options:
- **Git LFS**: `git lfs track "models/**/*.lgb"`
- **Cloud Storage**: S3, Google Cloud Storage
- **Streamlit Secrets**: Small models only

3. **Deploy on Streamlit Cloud**

- Go to https://share.streamlit.io
- Click "New app"
- Connect GitHub repository
- Select `app.py`
- Click "Deploy"

4. **Configure Secrets** (if needed)

In Streamlit Cloud dashboard:
- Settings → Secrets
- Add any API keys or credentials

### Model Loading Strategy

**Option A**: Download models on first run
```python
@st.cache_resource
def download_and_load_models():
    # Download from S3/GCS
    # Load into memory
    # Cache for subsequent requests
```

**Option B**: Include in deployment
- Use Git LFS for model files
- Streamlit Cloud supports LFS

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
# Build image
docker build -t ag-iq-valuation .

# Run container
docker run -p 8501:8501 ag-iq-valuation

# Access at http://localhost:8501
```

### Docker Compose

```yaml
version: '3.8'

services:
  ag-iq:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data:ro
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

---

## Cloud Platform Deployment

### AWS (EC2 + ECS)

**Option 1: EC2 Instance**

```bash
# SSH into EC2
ssh -i key.pem ubuntu@ec2-instance

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv libomp-dev

# Clone repo
git clone <repo-url>
cd ag_iq_ml

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run with systemd
sudo systemctl start ag-iq
```

**Option 2: ECS (Container)**
- Push Docker image to ECR
- Create ECS task definition
- Deploy to Fargate or EC2

### Google Cloud (Cloud Run)

```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/ag-iq

# Deploy
gcloud run deploy ag-iq \
  --image gcr.io/PROJECT_ID/ag-iq \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure (App Service)

```bash
# Create App Service
az webapp create --resource-group RG --plan PLAN --name ag-iq

# Deploy
az webapp up --name ag-iq --runtime "PYTHON:3.10"
```

---

## Production Considerations

### Security

**Authentication** (if needed):
```python
import streamlit_authenticator as stauth

# Add to app.py
authenticator = stauth.Authenticate(...)
name, authentication_status, username = authenticator.login()

if authentication_status:
    # Show app
else:
    st.error("Please login")
```

**Rate Limiting**:
```python
# Limit predictions per user/IP
from streamlit_extras.app_utils import rate_limit

@rate_limit(max_calls=100, period=3600)
def predict():
    ...
```

### Monitoring

**Logging**:
```python
import logging

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# Log predictions
logging.info(f"Prediction: {category} {make} ${price}")
```

**Metrics**:
- Track prediction counts
- Monitor response times
- Log errors and exceptions
- Track model performance vs actuals

### Performance Optimization

**Caching**:
```python
@st.cache_resource  # Models (loaded once)
@st.cache_data      # Reference data (cached)
```

**Model Loading**:
- Load models on startup (not per request)
- Use lightweight models (<5MB each)
- Consider model quantization if needed

### Scaling

**Horizontal Scaling**:
- Multiple Streamlit instances behind load balancer
- Shared model storage (S3/GCS)
- Session affinity for caching

**Vertical Scaling**:
- 2-4 CPU cores sufficient
- 4-8GB RAM recommended
- SSD for faster model loading

---

## CI/CD Pipeline

### GitHub Actions Example

```yaml
name: CI/CD

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python -m pytest tests/
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to Streamlit Cloud
        run: |
          # Trigger deployment
```

---

## Model Retraining Strategy

### Frequency

**Recommended**: Monthly or quarterly

**Triggers**:
- New data available (monthly auctions)
- Performance degradation detected
- Market conditions change significantly
- New features added

### Retraining Process

```bash
# 1. Update raw data
cp new_data/*.xlsx data/raw/

# 2. Retrain models
python run_complete_pipeline.py  # Full pipeline
# OR
python train_log_models.py       # Just retrain models

# 3. Validate performance
python -c "from src.models.fmv_log import FMVLogModel; ..."

# 4. Deploy updated models
# Copy to production server or trigger CD pipeline
```

### Version Control

```bash
# Tag model versions
git tag -a v1.0.0 -m "Initial production models"
git push origin v1.0.0

# Track model performance
models/
  fmv_harvesting_log/
    metadata.json  # Contains training date, metrics
```

---

## Data Management

### Data Storage

**Development**:
- Local files in `data/raw/`
- Parquet files in `data/processed/`

**Production**:
- S3/GCS for raw data
- Versioned datasets
- Automated backups

### Data Updates

```bash
# Monthly data refresh
1. Export new auction data from database
2. Copy to data/raw/
3. Run: python run_complete_pipeline.py
4. Retrain: python train_log_models.py
5. Validate: Test in Streamlit
6. Deploy: Push updated models
```

---

## Troubleshooting

### Common Deployment Issues

**Issue**: Models not loading
```bash
# Check model files exist
ls -la models/fmv_*_log/

# Verify metadata
cat models/fmv_harvesting_log/metadata.json
```

**Issue**: Out of memory
```bash
# Reduce model size or increase RAM
# Use model quantization
# Load models lazily
```

**Issue**: Slow predictions
```bash
# Check caching is working
# Verify models are cached (@st.cache_resource)
# Consider model optimization
```

### Performance Monitoring

```python
import time

start = time.time()
prediction = model.predict(data)
duration = time.time() - start

if duration > 2.0:
    logging.warning(f"Slow prediction: {duration:.2f}s")
```

---

## Security Best Practices

1. **Don't commit**:
   - API keys
   - Database credentials
   - Raw data files
   - Large model files

2. **Use environment variables**:
```python
import os
API_KEY = os.getenv('API_KEY')
```

3. **Validate inputs**:
```python
if year < 1980 or year > 2026:
    raise ValueError("Invalid year")
```

4. **Rate limiting**:
- Prevent abuse
- Limit API calls per user

5. **HTTPS only**:
- SSL certificate required
- No unencrypted data transmission

---

## Support

### Monitoring Checklist

- [ ] Application uptime
- [ ] Prediction latency
- [ ] Error rates
- [ ] Model performance vs actuals
- [ ] User feedback

### Maintenance Schedule

- **Daily**: Check error logs
- **Weekly**: Review prediction accuracy
- **Monthly**: Retrain models with new data
- **Quarterly**: Performance review, feature additions

---

For deployment questions, contact the development team.
