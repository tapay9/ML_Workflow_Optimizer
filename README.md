# Project Title: ModelPilot

> ModelPilot is an AI-powered machine learning workflow optimization platform designed to streamline model experimentation, deployment, real-time monitoring, experiment tracking, and hyperparameter optimization. Built with Python, Flask, D3.js, scikit-learn, SQLite, and the Anthropic API, ModelPilot provides an integrated environment where machine learning engineers can train, evaluate, analyze, deploy, and monitor models. The project uses hotel booking cancellation prediction as a real-world use case to demonstrate the complete machine learning workflow.

---

## 👥 Team Members & Contributions

As per course requirements, all three members actively collaborated and committed to this repository.

* **Taniyah Payton (tapay9)** - Data Engineering, AI Integration & Application Development
* **Isaac Rankin (themaxhat)** - Model Training, Experiment Tracking & Monitoring
* **Tariq Lukis (taluk1)** - Model Evaluation, Optimization & Deployment

---

## 🚀 Live Demo

* **Application URL:** [ModelPilot Live Application](https://ml-workflow-optimizer-git-548146111904.us-east1.run.app/)
* *Note: This live demo will be fully functional and accessible during the live project presentation.*

### 🎥 Project Walkthrough

![ModelPilot Project Walkthrough](demo.gif)

---

## 🛠️ Installation & Setup

Follow these steps to set up the environment and run the project locally.

### Prerequisites

* Python 3.10+
* `pip`
* Git
* Anthropic API key

### Local Installation

```bash
# Clone the repository
git clone https://github.com/tapay9/ML_Workflow_Optimizer.git
cd ML_Workflow_Optimizer

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment on macOS/Linux
source venv/bin/activate

# On Windows use:
# venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Configure the Anthropic API

Create a `.env` file in the root directory:

```env
ANTHROPIC_API_KEY=your_api_key_here
```

The `.env` file contains sensitive credentials and should never be committed to GitHub.

### Run ModelPilot

```bash
python3 app/app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## 📊 Dataset

* **Source:** [Kaggle - Hotel Booking Reservation Dataset](https://www.kaggle.com/datasets/kundanbedmutha/hotel-booking-reservation)
* **Description:** The dataset contains hotel booking information used to predict whether a reservation will be canceled. The target variable is `is_canceled`, where `0` represents a reservation that was not canceled and `1` represents a canceled reservation. The raw dataset contained 119,390 records and 33 columns. After preprocessing, the cleaned dataset contains 119,144 records and 29 columns.
* **Data Pipeline:** The processed dataset used for model training is stored in `data/hotel_bookings_cleaned.csv`.

### Data Preprocessing

The preprocessing pipeline includes:

* Handling missing values
* Removing duplicate records
* Removing invalid bookings
* Removing leakage-related fields
* Cleaning categorical values
* Preparing features for model training
* Feature engineering

Engineered features include:

* `total_guests`
* `total_nights`
* `hotel_type`

Leakage-related information, including reservation status information, was removed before model training.

---

## 🧠 Model Architecture & Training

* **Algorithms:** Logistic Regression, Decision Tree, and K-Nearest Neighbors (KNN)
* **Frameworks Used:** scikit-learn, pandas, NumPy
* **Training Process:** Models were trained using the preprocessed hotel booking dataset and evaluated across multiple hyperparameter configurations, training-data sizes, and train/test splits. Model performance was compared using test accuracy, Macro F1-score, ROC-AUC, recall, training accuracy, and the train/test accuracy gap.

### Logistic Regression

Logistic Regression was used as a linear classification baseline and demonstrated stable generalization across training-data experiments.

### Decision Tree

Decision Tree models were tuned using different hyperparameters to improve predictive performance while reducing overfitting. The final baseline Decision Tree used `max_depth=12`.

### K-Nearest Neighbors

KNN was evaluated using distance-based classification. The final KNN experiment used `k=41`, distance weighting, and Manhattan distance (`p=1`).

---

## 📈 Evaluation & Results

The final baseline model versions used by ModelPilot are v7, v8, and v9.

Earlier versions were created during development, training, and tuning experiments. ModelPilot maintains model version history instead of overwriting previous experiments.

| Model | Version | Test Accuracy | Macro F1 | ROC-AUC | Cancellation Recall | Train/Test Gap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | v7 | 81.18% | 0.7883 | 0.8883 | 64.57% | -0.18% |
| Decision Tree | v8 | **83.22%** | **0.8176** | **0.9100** | **74.10%** | 1.86% |
| K-Nearest Neighbors | v9 | 80.68% | 0.7801 | 0.8789 | 61.85% | 19.32% |

Decision Tree v8 produced the strongest overall baseline test performance.

KNN v9 achieved perfect training performance during the final experiment but had a much larger train/test gap, indicating substantial overfitting.

Evaluation and analysis outputs generated during development include:

* Confusion matrices
* ROC curves
* Precision-recall curves
* Feature significance analysis
* Model comparison results
* Experiment results

These analysis outputs are stored under:

```text
src/analysis_output/
```

---

## 🧑‍💻 Git Workflow & Guidelines

To maintain collaboration integrity:

1. **Branch Frequently:** Create feature branches for new project features.

```bash
git switch -c feature/your-feature
```

2. **Commit Regularly:** Write descriptive commit messages.

```bash
git add <specific-files>
git commit -m "Describe completed feature"
```

3. **Pull Before Push:** Pull the latest changes before beginning or merging work.

```bash
git pull origin main
```

4. **Push Feature Branches:**

```bash
git push origin feature/your-feature
```

5. Open a pull request and merge reviewed work into `main`.

Sensitive files such as `.env` should never be committed to the repository.

---

# Additional ModelPilot Documentation

## ✨ Key Features

ModelPilot extends the required machine learning workflow with an integrated web interface for model management, AI-assisted optimization, deployment, prediction, and monitoring.

Major features include:

* Model registry and version management
* Automated experiment tracking
* AI-powered model analysis
* Hyperparameter recommendations
* AI-assisted model retraining
* Controlled Production deployment
* Live prediction serving
* D3.js model monitoring
* Cloud deployment
* Jupyter Notebook documentation

---

## 🤖 Anthropic AI Insights

ModelPilot integrates the Anthropic API to analyze machine learning results and provide intelligent recommendations.

The AI Insights system provides:

* Model performance analysis
* Overfitting observations
* Optimization recommendations
* Hyperparameter suggestions
* Deployment-readiness feedback
* Recommended next experiments

The AI acts as a decision-support system while keeping model training and deployment decisions under engineer control.

---

## 🎯 AI-Assisted Hyperparameter Optimization

Anthropic analyzes model performance and recommends specific model configurations for further experimentation.

Engineers can select **Accept & Retrain** to:

1. Accept the AI-generated hyperparameter recommendation.
2. Send the configuration to a Flask endpoint.
3. Validate the recommended model and parameters.
4. Retrain the selected model.
5. Evaluate the candidate on the test dataset.
6. Register the result as a new model version.

AI-generated candidates are not automatically promoted to Production. This allows engineers to evaluate and compare their performance before making a deployment decision.

---

## 🤖 AI-Assisted Retraining

The AI-assisted optimization workflow follows:

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

During testing, Anthropic recommended retraining the Decision Tree using:

```text
max_depth = 10
min_samples_leaf = 4
criterion = entropy
```

Accepting this recommendation triggered the retraining workflow and produced **v10** as a new candidate model.

### Decision Tree v8 vs. AI-Retrained v10

| Metric | Decision Tree v8 | AI-Retrained v10 |
| :--- | :---: | :---: |
| Test Accuracy | **83.22%** | 82.84% |
| Macro F1 | **0.8176** | 0.8157 |
| ROC-AUC | **0.9100** | 0.9093 |
| Train/Test Gap | 1.86% | **0.41%** |

The AI-generated configuration substantially reduced the train/test gap while maintaining performance close to the original Decision Tree.

However, v10 did not outperform v8 across every test metric. ModelPilot therefore treats AI-retrained models as candidates for evaluation rather than automatically replacing an existing Production model.

This demonstrates a human-in-the-loop approach to AI-assisted machine learning optimization.

---

## 🗄️ Model Registry & Versioning

ModelPilot uses a dedicated SQLite database for model metadata, versioning, and deployment management.

The database is located at:

```text
data/mlwo.db
```

Model artifacts are stored under:

```text
models/hotel_cancellation/
```

Each registered version can contain:

* Algorithm information
* Model metrics
* Feature information
* Model parameters
* Model artifact
* Deployment stage
* Experiment notes

This allows ModelPilot to maintain a history of trained models rather than overwriting previous experiments.

---

## 🧪 Automated Experiment Tracking

ModelPilot automatically records machine learning experiments using Python.

Experiment records include:

* Algorithm
* Hyperparameters
* Evaluation metrics
* Experiment metadata

Experiment information is stored in:

```text
src/analysis_output/experiment_log.json
```

This allows engineers to compare experiments without manually maintaining separate records.

---

## 🚀 Model Deployment

Model training and Production deployment are intentionally separate processes.

The ModelPilot deployment workflow is:

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

The Deployment interface allows engineers to:

* Review registered model versions
* Compare model metrics
* Promote approved versions
* Change the Production model
* Roll back deployments
* Run predictions against registered models

Only one model version is designated as Production at a time.

This prevents newly trained or AI-generated models from automatically replacing an existing Production model without evaluation.

---

## 🔮 Prediction Demo

The Demo interface allows users to run hotel cancellation predictions using registered models.

Prediction requests are sent to the Flask backend, which loads the appropriate model and generates an inference result.

This demonstrates that registered models are not only stored for comparison but can also be used for real predictions.

Prediction activity is recorded and can later be viewed through the Monitoring dashboard.

---

## 📡 Real-Time Model Monitoring

ModelPilot uses D3.js and Python-generated monitoring data to visualize model activity.

The monitoring workflow is:

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

The Monitoring dashboard displays information such as:

* Prediction activity
* Active model versions
* Deployment stage
* Prediction latency
* Model usage

Monitoring information is generated from actual prediction activity.

---

## 🧭 ModelPilot Application Pages

### Dashboard

Provides an overview of the ModelPilot machine learning workflow and major system components.

### Models

Displays final model performance, configurations, model selection information, and registered model versions.

### Experiments

Displays tracked machine learning experiments and their results.

### Monitoring

Uses D3.js visualizations to display model prediction activity and monitoring information.

### AI Insights

Uses Anthropic to provide:

* Performance analysis
* Optimization suggestions
* Hyperparameter recommendations
* Deployment-readiness feedback
* AI-assisted retraining

### Deployment

Provides model version and Production deployment controls.

### Demo

Allows users to interact with the model prediction workflow and generate hotel cancellation predictions.

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
Build Container Image
     ↓
Artifact Registry
     ↓
Google Cloud Run
     ↓
Serve ModelPilot
```

The application is packaged using Docker and served using Gunicorn.

Application secrets such as the Anthropic API key are configured through the deployment environment and are not stored in the public source code.

### Cloud Deployment Limitation

The current SQLite database uses the Cloud Run instance's local filesystem. This storage is not persistent across all Cloud Run instance lifecycle events.

A future production implementation could migrate the model registry to a persistent database such as Google Cloud SQL or Firestore.

---

## 📚 Jupyter Notebook Documentation

A complete ModelPilot user tutorial is available at:

```text
notebooks/ModelPilot_User_Tutorial.ipynb
```

The tutorial explains the application's major features and walks users through the complete ModelPilot workflow.

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

## 🏗️ Technology Stack

| Component | Technology |
| :--- | :--- |
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

COSC 472 — Machine Learning
