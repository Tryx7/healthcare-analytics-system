
# 🚀 Deployment Guide

## Quick Deploy to Vercel (Recommended)

### Step 1: Prepare Your Repository
```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit"

# Create GitHub repo and push
git remote add origin https://github.com/YOUR_USERNAME/healthcare-analytics.git
git push -u origin main
```

### Step 2: Deploy to Vercel

#### Option A: Vercel CLI
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

#### Option B: Vercel Dashboard
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repository
3. Select "Python" framework
4. Set environment variables (if needed)
5. Deploy!

### Step 3: Verify Deployment
```bash
# Test health endpoint
curl https://your-app.vercel.app/health

# Test prediction
curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "Male",
    "blood_type": "O+",
    "medical_condition": "Diabetes",
    "admission_type": "Emergency",
    "billing_amount": 15000,
    "length_of_stay": 3,
    "age_group": "Adult",
    "billing_category": "Medium",
    "medication": "Metformin"
  }'
```

## Deploy to Render

1. Go to [render.com](https://render.com)
2. Create New Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn api:app`
6. Add environment variables
7. Deploy

## Deploy to Railway

1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub repo
3. Add PostgreSQL service (optional)
4. Set environment variables
5. Deploy

## Local Development with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Environment Variables

Create a `.env` file:
```
# Aiven PostgreSQL (pre-configured)
DB_HOST=pg-c63647-lagatkjosiah-692c.c.aivencloud.com
DB_PORT=24862
DB_NAME=defaultdb
DB_USER=avnadmin
DB_PASSWORD=AVNS_G1ajzCj_WUpXrLzc-3t
DB_SSL_MODE=require

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

## API Usage Examples

### cURL
```bash
# Health check
curl https://your-app.vercel.app/health

# Single prediction
curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45,
    "gender": "Male",
    "blood_type": "O+",
    "medical_condition": "Diabetes",
    "admission_type": "Emergency",
    "billing_amount": 15000,
    "length_of_stay": 3,
    "age_group": "Adult",
    "billing_category": "Medium",
    "medication": "Metformin"
  }'

# Batch prediction
curl -X POST https://your-app.vercel.app/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "patients": [
      {"age": 45, "gender": "Male", ...},
      {"age": 30, "gender": "Female", ...}
    ]
  }'
```

### Python
```python
import requests

# Health check
response = requests.get("https://your-app.vercel.app/health")
print(response.json())

# Prediction
payload = {
    "age": 45,
    "gender": "Male",
    "blood_type": "O+",
    "medical_condition": "Diabetes",
    "admission_type": "Emergency",
    "billing_amount": 15000,
    "length_of_stay": 3,
    "age_group": "Adult",
    "billing_category": "Medium",
    "medication": "Metformin"
}
response = requests.post("https://your-app.vercel.app/predict", json=payload)
print(response.json())
```

### JavaScript
```javascript
// Prediction
fetch('https://your-app.vercel.app/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    age: 45,
    gender: 'Male',
    blood_type: 'O+',
    medical_condition: 'Diabetes',
    admission_type: 'Emergency',
    billing_amount: 15000,
    length_of_stay: 3,
    age_group: 'Adult',
    billing_category: 'Medium',
    medication: 'Metformin'
  })
})
.then(res => res.json())
.then(data => console.log(data));
```
