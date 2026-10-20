# ModelPilot

> **ModelPilot** is an AI-powered ML workflow optimization platform designed to streamline model deployment, real-time monitoring, experiment tracking, and hyperparameter optimization. Built with Python, Flask, D3.js, scikit-learn, SQLite, and the Anthropic API, ModelPilot provides an integrated environment where machine learning engineers can train, evaluate, analyze, deploy, and monitor models. The project uses hotel booking cancellation prediction as a real-world use case to demonstrate the complete machine learning workflow.

## 👥 Team Members & Contributions

As per course requirements, all three members actively collaborated and committed to this repository.

- **Taniyah Payton (tapay9)** — Data Engineering, AI Integration & Application Development
- **Isaac Rankin (themaxhat)** — Model Training, Experiment Tracking & Monitoring
- **Tariq Lukis (taluk1)** — Model Evaluation, Optimization & Deployment

---

## 🚀 Live Demo

ModelPilot is deployed on **Google Cloud Run** and is accessible through a web browser.

- **Application:** ModelPilot
- **Cloud Platform:** Google Cloud Run
- **Deployment:** Production cloud deployment

> Add the final public Cloud Run URL here before submission.

### 🎥 Project Walkthrough

A project walkthrough demonstrates the complete ModelPilot workflow, including model comparison, experiment tracking, AI-generated insights, deployment management, predictions, and real-time monitoring.

> Add `demo.gif` here before final submission if required by the course.

---

## ✨ Key Features

### 🤖 Anthropic AI Insights

ModelPilot integrates the **Anthropic API** to analyze machine learning results and provide intelligent recommendations.

The AI Insights system provides:

- Model performance analysis
- Overfitting observations
- Optimization recommendations
- Hyperparameter suggestions
- Deployment-readiness feedback
- Recommended next experiments

The AI acts as a decision-support system while keeping model training and deployment decisions under engineer control.

### 🎯 AI-Assisted Hyperparameter Optimization

Anthropic analyzes model performance and recommends a specific model configuration for further experimentation.

Engineers can select **Accept & Retrain** to:

1. Accept the AI-generated hyperparameter recommendation.
2. Send the configuration to a Flask endpoint.
3. Validate the recommended model and parameters.
4. Retrain the selected model.
5. Evaluate the candidate on the test dataset.
6. Register the result as a new model version.

AI-generated candidates are not automatically promoted to Production. This allows engineers to compare their performance before making a deployment decision.

### 📊 Real-Time Model Monitoring

ModelPilot uses **D3.js** and Python-generated monitoring data to visualize model activity.

The Monitoring dashboard displays information such as:

- Prediction activity
- Active model versions
- Deployment stage
- Prediction latency
- Model usage

Monitoring data is generated from actual prediction activity.

### 🧪 Automated Experiment Tracking

ModelPilot automatically records machine learning experiments using Python.

Experiment records include:

- Algorithm
- Hyperparameters
- Evaluation metrics
- Experiment metadata

Experiment information is stored in JSON format and displayed through the **Experiments** interface for comparison and analysis.

### 🚀 Model Deployment

ModelPilot includes model versioning and deployment controls backed by SQLite.

Engineers can:

- Review registered model versions
- Compare model metrics
- Promote approved versions
- Change the Production model
- Roll back deployments
- Run predictions against registered models

Only one model version is designated as Production at a time.

---

## 🛠️ Installation & Setup

Follow these steps to run ModelPilot locally.

### Prerequisites

- Python 3.10+
- `pip`
- Git
- Anthropic API key

### 1. Clone the Repository

```bash
git clone https://github.com/tapay9/ML_Workflow_Optimizer.git
cd ML_Workflow_Optimizer
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Anthropic API

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

The `.env` file contains sensitive credentials and should **never be committed to GitHub**.

### 5. Run ModelPilot

```bash
python3 app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 📊 Dataset

### Hotel Booking Cancellation Dataset

ModelPilot uses hotel booking data to demonstrate its machine learning workflow.

- **Source:** Kaggle — Hotel Booking Reservation Dataset
- **Task:** Binary classification
- **Target:** `is_canceled`
- `0` = reservation was not canceled
- `1` = reservation was canceled

After preprocessing, the cleaned dataset contains approximately **119,144 records and 29 columns**.

The processed dataset is stored at:

```text
data/hotel_bookings_cleaned.csv
```

### Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Removing duplicates
- Removing leakage-related fields
- Cleaning categorical values
- Preparing data for model training
- Feature engineering

Engineered features include:

- `total_guests`
- `total_nights`
- `hotel_type`

Leakage-related information such as reservation status information is removed before model training.

---

## 🧠 Model Architecture & Training

ModelPilot evaluates three classification algorithms:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

### Technologies

Model development uses:

- Python
- pandas
- NumPy
- scikit-learn

Models are evaluated using:

- Test Accuracy
- Macro F1-score
- ROC-AUC
- Recall
- Training Accuracy
- Train/Test Accuracy Gap

The train/test gap is used as an additional indicator of potential overfitting.

---

## 📈 Final Model Evaluation

The final baseline model versions used by ModelPilot are **v7, v8, and v9**.

| Model | Version | Test Accuracy | Macro F1 | ROC-AUC | Train/Test Gap |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | v7 | 81.18% | 0.7883 | 0.8883 | -0.18% |
| Decision Tree | v8 | **83.22%** | **0.8176** | **0.9100** | 1.86% |
| K-Nearest Neighbors | v9 | 80.68% | 0.7801 | 0.8789 | 19.32% |

Decision Tree v8 produced the strongest overall baseline test performance.

KNN v9 achieved perfect training performance during the final experiment but had a much larger train/test gap, indicating substantial overfitting.

These results are provided to the Anthropic-powered AI Insights system for further analysis and optimization recommendations.

---

## 🤖 AI-Assisted Retraining

ModelPilot connects AI-generated recommendations directly to the machine learning training workflow.

The optimization process follows:

```text
Model Results
      ↓
Anthropic AI Analysis
      ↓
Hyperparameter Recommendation
      ↓
Engineer Review
      ↓
Accept & Retrain
      ↓
Flask Retraining Endpoint
      ↓
Model Training
      ↓
Evaluation
      ↓
New Candidate Version
```

### Example AI Recommendation

During testing, Anthropic recommended retraining the Decision Tree using parameters including:

```text
max_depth = 10
min_samples_leaf = 4
criterion = entropy
```

Accepting the recommendation triggered the retraining workflow and produced **v10** as a new local candidate model.

### v8 vs. AI-Retrained v10

| Metric | Decision Tree v8 | AI-Retrained v10 |
|---|---:|---:|
| Test Accuracy | **83.22%** | 82.84% |
| Macro F1 | **0.8176** | 0.8157 |
| ROC-AUC | **0.9100** | 0.9093 |
| Train/Test Gap | 1.86% | **0.41%** |

The AI-generated configuration substantially reduced the train/test gap while maintaining performance close to the original Decision Tree.

However, v10 did not outperform v8 across every test metric. ModelPilot therefore treats AI-retrained models as **candidates for evaluation** rather than automatically replacing the existing Production model.

This demonstrates a human-in-the-loop approach to AI-assisted ML optimization.

---

## 🗄️ Model Registry & Versioning

ModelPilot uses a dedicated **SQLite database** for model metadata, versioning, and deployment management.

The database is located at:

```text
data/mlwo.db
```

Model artifacts are stored under:

```text
models/hotel_cancellation/
```

Each registered version can contain:

- Algorithm information
- Model metrics
- Feature information
- Model parameters
- Model artifact
- Deployment stage
- Experiment notes

This allows ModelPilot to maintain a history of trained models rather than overwriting previous experiments.

---

## 🚀 Deployment Workflow

Model training and Production deployment are intentionally separate processes.

A typical ModelPilot deployment workflow is:

```text
Train
  ↓
Evaluate
  ↓
Register Candidate
  ↓
Compare
  ↓
Deployment Decision
  ↓
Production
```

This prevents newly trained or AI-generated models from automatically replacing an existing Production model without evaluation.

The **Deployment** interface allows engineers to manage model versions and control which approved version is used in Production.

---

## 🔮 Prediction Demo

The **Demo** interface allows users to run hotel cancellation predictions using registered models.

Prediction requests are sent to the Flask backend, which loads the appropriate model and generates an inference result.

This demonstrates that registered models are not only stored for comparison—they can also be used for real predictions.

Prediction activity is recorded and can later be viewed through the Monitoring dashboard.

---

## 📡 Real-Time Monitoring

ModelPilot connects model serving with monitoring.

The workflow is:

```text
Deploy Model
     ↓
Run Prediction
     ↓
Record Prediction Activity
     ↓
Update Monitoring Data
     ↓
Visualize with D3.js
```

The Monitoring page provides an interactive view of model activity and helps engineers understand how deployed models are being used.

---

## 🧪 Experiment Tracking

ModelPilot automatically records experiments using Python.

Experiment information is stored in:

```text
src/analysis_output/experiment_log.json
```

The experiment-tracking workflow records model configurations, metrics, and metadata so engineers can compare results without manually maintaining experiment records.

Additional analysis outputs are stored under:

```text
src/analysis_output/
```

These may include:

- Confusion matrices
- ROC curves
- Precision-recall curves
- Feature significance analysis
- Model comparison results
- Experiment results

---

## 🧭 ModelPilot Application Pages

### Dashboard

Provides an overview of the ModelPilot ML workflow and major system components.

### Models

Displays final model performance, configurations, model selection information, and registered model versions.

### Experiments

Displays tracked machine learning experiments and their results.

### Monitoring

Uses D3.js visualizations to display model prediction activity and monitoring information.

### AI Insights

Uses Anthropic to provide:

- Performance analysis
- Optimization suggestions
- Hyperparameter recommendations
- Deployment-readiness feedback
- AI-assisted retraining

### Deployment

Provides model version and Production deployment controls.

### Demo

Allows users to interact with the model prediction workflow.

---

## 🧪 Testing

ModelPilot includes automated tests for important application components and endpoints.

Run the test suite with:

```bash
python -m pytest tests
```

During final development, the project test suite completed successfully with:

```text
6 passed
```

Testing helps verify functionality across model serving, registry operations, Flask endpoints, and application workflows.

---

## ☁️ Cloud Deployment

ModelPilot is deployed using **Google Cloud Run**.

The cloud deployment allows the Flask application to be accessed through a public web interface without requiring users to configure the project locally.

The deployment workflow is:

```text
GitHub main
     ↓
Google Cloud Build
     ↓
Install Python Dependencies
     ↓
Create Cloud Run Revision
     ↓
Serve ModelPilot
```

Application secrets such as the Anthropic API key are configured through the deployment environment and are not stored in the public source code.

---

## 📚 Jupyter Notebook Documentation

A complete ModelPilot user tutorial is available at:

```text
notebooks/ModelPilot_User_Tutorial.ipynb
```

The tutorial explains the application's major features and walks engineers through the complete ModelPilot workflow.

Additional machine learning notebooks include:

```text
src/model_architecture.ipynb
src/model_experimentation.ipynb
src/model_analysis.ipynb
```

These notebooks document model development, experimentation, and analysis.

---

## 🔄 Complete ModelPilot Workflow

ModelPilot combines the major stages of the machine learning engineering lifecycle into one platform:

```text
Train
  ↓
Track
  ↓
Analyze
  ↓
Optimize
  ↓
Evaluate
  ↓
Deploy
  ↓
Predict
  ↓
Monitor
```

Rather than requiring engineers to manage model experiments, AI analysis, deployment, prediction, and monitoring through separate systems, ModelPilot provides these capabilities through one integrated interface.

---

## 🧑‍💻 Git Workflow & Guidelines

To maintain collaboration integrity:

1. **Create feature branches**

```bash
git switch -c feature/your-feature
```

2. **Commit completed work regularly**

```bash
git add <specific-files>
git commit -m "Describe completed feature"
```

3. **Pull the latest changes before beginning new work**

```bash
git pull origin main
```

4. **Push feature branches**

```bash
git push origin feature/your-feature
```

5. Open a pull request and merge reviewed work into `main`.

Sensitive files such as `.env` should never be committed to the repository.

---

## 🏗️ Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn |
| Data Processing | pandas, NumPy |
| AI Integration | Anthropic API |
| Visualization | D3.js |
| Model Registry | SQLite |
| Experiment Storage | JSON |
| Frontend | HTML, CSS, JavaScript |
| Testing | pytest / Python testing |
| Cloud Deployment | Google Cloud Run |
| Version Control | Git & GitHub |
| Documentation | Jupyter Notebook |

---

## 🎯 Project Purpose

ModelPilot demonstrates how AI can support the machine learning engineering lifecycle beyond simply generating predictions.

By combining model experimentation, version management, AI-generated insights, hyperparameter recommendations, controlled deployment, prediction serving, and real-time monitoring, ModelPilot provides a unified workflow for managing machine learning models.

The platform uses Anthropic as an intelligent assistant while keeping evaluation and deployment decisions under engineer control.

---

## 👥 Team Squarepants

**Taniyah Payton · Isaac Rankin · Tariq Lukis**

**COSC 472 — Machine Learning**
