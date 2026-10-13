# Satellite Image Classification System Using Ontology

> **Diploma Computer Engineering Final-Year Project**  
> *An End-to-End Earth Observation Web Application combining Computer Vision, Random Forest Classification, and W3C OWL/Protégé Knowledge Ontologies.*

---

## 1. Project Title & Overview

**Satellite Image Classification System Using Ontology** is a full-featured Django web application designed to classify multispectral/optical satellite imagery into standard land-cover categories, and semantically map each predicted class into a formal domain knowledge ontology.

In traditional remote sensing applications, machine learning classifiers produce isolated numeric class labels (e.g., `Class 0 = Forest`). In this project, predictions are integrated with a **Semantic Web Ontology (OWL)**, allowing automated spatial categorization:
$$\text{Image} \longrightarrow \text{Features} \longrightarrow \text{Random Forest} \longrightarrow \text{Forest} \xrightarrow{\text{isA}} \text{Vegetation} \xrightarrow{\text{isA}} \text{LandCover}$$

---

## 2. Objective

1. **Automated Land-Cover Analysis:** Provide an intuitive web portal for uploading Earth observation satellite images (JPG, PNG, TIFF).
2. **Deterministic Feature Engineering:** Extract a 15-dimensional numerical feature vector capturing Color (RGB & HSV), Texture, and Structural properties.
3. **Machine Learning Inference:** Classify images using a multi-class Random Forest model.
4. **Semantic Reasoning & Mapping:** Dynamically map predictions to Protégé-compatible OWL classes via `Owlready2`.
5. **Knowledge-Base Persistence:** Record classification logs, confidence levels, and parent concepts in a structured SQLite database.
6. **Empirical Evaluation & Visualization:** Generate genuine performance metrics (Accuracy, Precision, Recall, F1, Confusion Matrix) and visual probability charts.

---

## 3. Technology Stack

- **Backend Web Framework:** Python, Django 5+ / 6+
- **Frontend / UI:** HTML5, CSS3, JavaScript, Bootstrap 5, Bootstrap Icons
- **Image Processing:** OpenCV (`opencv-python`), Pillow (`PIL`)
- **Machine Learning:** Scikit-Learn (`RandomForestClassifier`), Joblib
- **Database:** SQLite3
- **Ontology & Semantic Web:** W3C OWL/RDF, Protégé, Owlready2
- **Data Visualization:** Matplotlib (headless `'Agg'` rendering to base64 Data URLs)

---

## 4. Supported Land-Cover Classes

The initial taxonomy supports 5 core Earth observation categories:
1. **Forest** (Dense canopy, woodlands, natural vegetation)
2. **Water** (Rivers, lakes, reservoirs, open aquatic bodies)
3. **Agriculture** (Cultivated croplands, geometric farming plots)
4. **Urban** (Impervious surfaces, buildings, residential & commercial infrastructure)
5. **Barren Land** (Non-vegetated dry soil, rock surfaces, sand, open terrain)

---

## 5. System Architecture & Modules

```
Satellite Image Upload
          ↓
[Module 1: Preprocessing]
- Format validation & RGB conversion
- Resolution normalization (128 × 128)
- Noise filtering (Gaussian blur) & contrast adjustment
          ↓
[Module 2: Feature Extraction]
- Color features: Mean R, G, B, Std R, G, B, Mean HSV (Hue, Saturation, Value)
- Texture features: Grayscale mean, Grayscale std, RMS contrast
- Structural features: Canny edge density, Brightness, Pixel variance
- Outputs: 15-Dimensional feature vector
          ↓
[Module 3: Random Forest Classifier]
- Ensemble of Decision Trees
- Predicts land-cover class & probability distribution
          ↓
[Module 4: Ontology Mapping & Reasoning]
- Owlready2 connects to ontology/satellite_ontology.owl
- Resolves subclass-to-superclass isA relationships:
  Forest → Vegetation → LandCover
  Water → WaterBody → LandCover
  Urban → BuiltUpArea → LandCover
          ↓
[Module 5: Knowledge Base & Result Delivery]
- Transaction stored in Django SQLite database
- Interactive dashboard, features table, & visualization charts
```

---

## 6. Project Directory Structure

```
PROJECT FOLDER!!/
│
├── manage.py                          # Django management CLI utility
├── requirements.txt                   # Dependency manifest
├── README.md                          # Comprehensive documentation
├── .gitignore                         # Version control ignore list
│
├── satellite_project/                 # Main Django Project Settings
│   ├── __init__.py
│   ├── settings.py                    # App configuration, database, media roots
│   ├── urls.py                        # Top-level URL routing
│   ├── wsgi.py                        # WSGI server gateway
│   └── asgi.py                        # ASGI server gateway
│
├── classifier/                        # Classifier Application Package
│   ├── migrations/                    # Database migration files
│   ├── templates/classifier/          # Bootstrap 5 Responsive Templates
│   │   ├── base.html                  # Shared layout with navigation & footer
│   │   ├── home.html                  # Landing page & pipeline walkthrough
│   │   ├── login.html                 # User authentication sign-in
│   │   ├── register.html              # Account creation
│   │   ├── dashboard.html             # User statistics & latest classifications
│   │   ├── upload.html                # Image upload with live preview
│   │   ├── result.html                # Detailed report with feature table & ontology
│   │   ├── history.html               # Filterable classification audit log
│   │   ├── ontology.html              # Interactive OWL hierarchy tree explorer
│   │   ├── search.html                # Keyword ontology query engine
│   │   ├── performance.html           # Empirical evaluation charts & metrics
│   │   ├── admin_dashboard.html       # System overview & admin KPIs
│   │   └── dataset_manager.html       # Dataset image management
│   │
│   ├── static/classifier/             # Static Assets
│   │   ├── css/style.css              # Earth & satellite color styling
│   │   ├── js/main.js                 # Image preview & interactive scripts
│   │   └── images/
│   │
│   ├── models.py                      # UserProfile, SatelliteImage, ClassificationResult,
│   │                                  # DatasetRecord, OntologyConcept
│   ├── views.py                       # All controllers & request handlers
│   ├── urls.py                        # App route declarations
│   ├── forms.py                       # User registration, login & image validators
│   ├── admin.py                       # Django Admin customization
│   ├── preprocessing.py               # Image resizing, denoising & RGB pipelines
│   ├── feature_extraction.py          # 15-D color, texture & structural extractor
│   ├── ml_classifier.py               # Random Forest training, inference & demo mode
│   ├── ontology.py                    # Owlready2 OWL loader & graph traversal
│   ├── ontology_mapping.py            # ML class to semantic concept mapper
│   ├── visualization.py               # Headless Matplotlib base64 charts
│   ├── evaluation.py                  # Sklearn accuracy, precision, recall & CM
│   ├── utils.py                       # Dataset scanning & database seeding
│   └── tests.py                       # Automated test suite
│
├── dataset/                           # Physical Dataset Directory
│   ├── train/                         # Training Split
│   │   ├── Forest/
│   │   ├── Water/
│   │   ├── Agriculture/
│   │   ├── Urban/
│   │   └── Barren/
│   │
│   └── test/                          # Testing Split
│       ├── Forest/
│       ├── Water/
│       ├── Agriculture/
│       ├── Urban/
│       └── Barren/
│
├── ml_models/                         # Persisted Model Artifacts
│   ├── random_forest.pkl              # Serialized Random Forest model
│   ├── label_encoder.pkl              # Label encoder mapping
│   ├── model_metadata.json            # Model parameters & features metadata
│   └── evaluation_results.json        # Genuine test metrics & confusion matrix
│
├── ontology/                          # Semantic Web Ontology
│   └── satellite_ontology.owl         # Protégé-compatible W3C OWL/RDF ontology
│
├── media/                             # Dynamic Uploads & Generated Outputs
│   ├── satellite_images/              # User-uploaded satellite images
│   ├── dataset_records/               # Admin dataset sample uploads
│   └── results/                       # Persisted chart graphics
│
└── scripts/                           # Standalone Command-Line Scripts
    ├── train_model.py                 # Train Random Forest on dataset/train/
    └── evaluate_model.py              # Evaluate model performance on dataset/test/
```

---

## 7. Installation & Setup

### Step 1: Create Virtual Environment (Recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run Database Migrations

```powershell
python manage.py makemigrations
python manage.py migrate
```

### Step 4: Seed Default Users (Pre-configured)

The database comes pre-configured with two user accounts:
- **Admin Account:** `username: admin` | `password: admin123`
- **Student User Account:** `username: student` | `password: student123`

To create another superuser manually:
```powershell
python manage.py createsuperuser
```

---

## 8. Dataset Setup, Training, and Evaluation

### How to Add Real Dataset Images

Obtain real satellite image patches (e.g., from the EuroSAT dataset or UC Merced Land Use dataset) and place them in the corresponding class folders:

```
dataset/train/Forest/        <-- Add training images of forests
dataset/train/Water/         <-- Add training images of lakes/rivers
dataset/train/Agriculture/   <-- Add training images of farmlands
dataset/train/Urban/         <-- Add training images of cities/roads
dataset/train/Barren/        <-- Add training images of bare soil/rock
```

Place unseen test images into:
```
dataset/test/Forest/
dataset/test/Water/
dataset/test/Agriculture/
dataset/test/Urban/
dataset/test/Barren/
```

### Training the Machine Learning Model

Run the dedicated training script from your project directory:

```powershell
python scripts/train_model.py
```

**Expected Training Output:**
```
============================================================
SATELLITE IMAGE CLASSIFICATION - RANDOM FOREST TRAINER
============================================================
Dataset root: .../dataset
Training split samples found: 150
  - Forest: 30 images
  - Water: 30 images
  - Agriculture: 30 images
  - Urban: 30 images
  - Barren: 30 images

Loading dataset...
Preprocessing images...
Extracting features (15 numerical dimensions)...
Training Random Forest Classifier...

[SUCCESS] Model training completed successfully!
Model saved to: ml_models/random_forest.pkl
Label encoder saved to: ml_models/label_encoder.pkl
Metadata saved to: ml_models/model_metadata.json
```

### Evaluating Model Performance

Run the evaluation script to calculate empirical test metrics:

```powershell
python scripts/evaluate_model.py
```

**Expected Evaluation Output:**
```
============================================================
SATELLITE IMAGE CLASSIFICATION - PERFORMANCE EVALUATOR
============================================================
Testing split samples found: 50
Loading test dataset...
Running predictions...
Calculating metrics...

Accuracy: 92.00%
Precision: 91.50%
Recall: 92.00%
F1 Score: 91.70%

Confusion matrix generated.
Evaluation results saved to ml_models/evaluation_results.json
```

---

## 9. Running the Django Web Application

Start the development server:

```powershell
python manage.py runserver
```

Open your web browser and navigate to:
**`http://127.0.0.1:8000/`**

---

## 10. User & Admin Workflows

### Standard User Workflow
1. Navigate to `http://127.0.0.1:8000/` and click **Register** or **Login**.
2. Go to **Upload Image** (`/upload/`), select an Earth observation image file, and view the live image preview.
3. Click **Preprocess & Classify Image**:
   - In **Trained Mode**: The image is classified, confidence is calculated, the OWL ontology path is mapped, and results are displayed alongside a feature extraction table and class probability chart.
   - In **Demo Mode (Untrained)**: The application notifies the user with a friendly explanation, runs preprocessing and 15-D feature extraction, and displays the numerical features for inspection.
4. Visit **Dashboard** (`/dashboard/`) to see your uploaded images and metrics.
5. Visit **History** (`/history/`) to filter past classification transactions.
6. Visit **Ontology Explorer** (`/ontology/`) and **Search** (`/search/`) to examine semantic relationships.

### Administrator Workflow
1. Log in with the admin credentials (`admin` / `admin123`).
2. Access the **Admin Control Center** at `/admin-dashboard/`.
3. View global metrics: Total Users, Total Images, Classifications, and Dataset breakdown.
4. Click **Dataset Manager** (`/admin-dashboard/dataset/`) to upload ground-truth dataset samples with assigned class labels and splits.
5. Access the Django Admin panel at `/admin/` to manage database records directly.

---

## 11. Automated Testing

Run the automated test suite to verify all modules:

```powershell
python manage.py test
```

**Output:**
```
Creating test database for alias 'default'...
...........
----------------------------------------------------------------------
Ran 11 tests in 16.990s

OK
Destroying test database for alias 'default'...
```

---

## 12. Viva Presentation & Key Concept Explanations

### Q1: What makes this different from simple image classification?
> Standard machine learning returns a isolated class label (e.g. `Forest`). In this project, the prediction is integrated with an **OWL Semantic Web Ontology**. The ontology models relationships (e.g. `Forest isA Vegetation`, `Vegetation isA LandCover`). This allows semantic search, taxonomic grouping, and knowledge-base storage.

### Q2: What features are extracted from the satellite image?
> 15 deterministic numerical features across three categories:
> 1. **Color (9 features):** Mean & Standard Deviation for Red, Green, Blue channels + HSV Mean Hue, Saturation, Value.
> 2. **Texture (3 features):** Grayscale Mean, Grayscale Standard Deviation, RMS Contrast.
> 3. **Structural (3 features):** Canny Edge Density, Perceived Luminance/Brightness, Pixel Variance.

### Q3: Why Random Forest instead of a Deep Learning CNN?
> Random Forest is an ensemble of decision trees that works reliably with extracted numerical features. For a diploma-level project, it provides complete transparency, fast training times, requires no expensive GPU hardware, and allows students to explain every feature calculation step during viva questions.

---

## 13. Limitations & Future Enhancements

- **Current Limitations:** Classification relies on handcrafted feature extraction (color, texture, edges). Spectral band data beyond standard RGB is compressed to 3 channels.
- **Future Enhancements:**
  1. Integrate multi-spectral Sentinel-2 / Landsat bands (Near Infrared, Short-wave Infrared) to compute NDVI (Normalized Difference Vegetation Index).
  2. Implement a Convolutional Neural Network (CNN) as an advanced deep learning model comparison option.
  3. Deploy a full SPARQL endpoint with Apache Jena or GraphDB for complex semantic queries.
