
# 🏥 Healthcare Analytics System

A complete machine learning-powered healthcare analytics system that predicts patient test results (Normal, Abnormal, Inconclusive) using patient demographic and medical data.

## 📋 Features

- **Data Cleaning & Preprocessing**: Automated data cleaning pipeline
- **PostgreSQL Database**: Robust data storage with schema optimization
- **Machine Learning**: Random Forest classifier for test result prediction
- **Automated Retraining**: Weekly model retraining every Saturday at 12:00 PM
- **REST API**: FastAPI-based API for predictions and data management
- **Batch Predictions**: Support for single and batch predictions
- **Model Monitoring**: Track model performance and versions

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Data Cleaning   │────▶│   PostgreSQL    │
│   (CSV)         │     │   & Validation   │     │   Database      │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                    ┌─────────────────────────────────────┘
                    ▼
           ┌─────────────────┐     ┌──────────────────┐
           │   ML Training   │◀────│  Scheduler       │
           │   (Weekly)      │     │  (Saturday 12PM) │
           └────────┬────────┘     └──────────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │   FastAPI       │
           │   REST API      │
           └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/healthcare-analytics.git
   cd healthcare-analytics
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

5. **Initialize the system**
   ```bash
   python initialize.py
   ```

6. **Start the API**
   ```bash
   python -m uvicorn main:app --reload
   ```

7. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Setup (Recommended)

```bash
# Start all services
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Predict Test Result
```bash
POST /predict
Content-Type: application/json

{
  "age": 45,
  "gender": "Male",
  "blood_type": "O+",
  "medical_condition": "Diabetes",
  "admission_type": "Emergency",
  "billing_amount": 15000.00,
  "length_of_stay": 3,
  "age_group": "Adult",
  "billing_category": "Medium",
  "medication": "Metformin"
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "patients": [
    { ...patient data... },
    { ...patient data... }
  ]
}
```

### Add Patient Record
```bash
POST /patients
Content-Type: application/json

{
  "name": "John Doe",
  "age": 45,
  "gender": "Male",
  "blood_type": "O+",
  "medical_condition": "Diabetes",
  "date_of_admission": "2024-01-15",
  "doctor": "Dr. Smith",
  "hospital": "General Hospital",
  "insurance_provider": "Blue Cross",
  "billing_amount": 15000.00,
  "room_number": 205,
  "admission_type": "Emergency",
  "discharge_date": "2024-01-18",
  "medication": "Metformin"
}
```

### Get Model Info
```bash
GET /model/info
```

### Trigger Retraining
```bash
POST /model/retrain?model_type=random_forest
```

### Get Statistics
```bash
GET /statistics
```

## 🔄 Automated Retraining

The system automatically retrains the ML model every **Saturday at 12:00 PM** using the APScheduler library. The retraining process:

1. Fetches latest data from PostgreSQL
2. Trains a new model with updated data
3. Evaluates model performance
4. Saves model metadata to database
5. Activates the new model for predictions

You can also manually trigger retraining via the `/model/retrain` endpoint.

## 🗄️ Database Schema

### patient_records
- Stores all patient healthcare records
- Includes derived features (length_of_stay, age_group, billing_category)

### predictions
- Stores model predictions with confidence scores
- Links to patient records

### model_metadata
- Tracks model versions and performance metrics
- Identifies active model for predictions

## 📊 Model Performance

The system uses a Random Forest classifier optimized for:
- **Accuracy**: Overall prediction correctness
- **Precision**: Minimizing false positives
- **Recall**: Minimizing false negatives
- **F1 Score**: Balanced measure of precision and recall

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test API endpoints
curl -X GET "http://localhost:8000/health"
```

## 🚢 Deployment

### Vercel Deployment

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Create vercel.json**
   ```json
   {
     "builds": [
       {
         "src": "main.py",
         "use": "@vercel/python"
       }
     ],
     "routes": [
       {
         "src": "/(.*)",
         "dest": "main.py"
       }
     ]
   }
   ```

3. **Deploy**
   ```bash
   vercel --prod
   ```

### Railway/Render Deployment

1. Connect your GitHub repository
2. Add PostgreSQL database service
3. Set environment variables
4. Deploy automatically on push

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| DB_HOST | PostgreSQL host | localhost |
| DB_PORT | PostgreSQL port | 5432 |
| DB_NAME | Database name | healthcare_db |
| DB_USER | Database user | postgres |
| DB_PASSWORD | Database password | password |
| API_HOST | API host | 0.0.0.0 |
| API_PORT | API port | 8000 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Healthcare Dataset by Prasad Patil on Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
- Built with FastAPI, scikit-learn, and PostgreSQL
