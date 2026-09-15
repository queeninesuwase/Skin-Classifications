
# Skin Lesion Classification
### Machine Learning — Medical Imaging Deep Learning Project

This repository contains a deep learning application engineered to categorize skin lesions from high-resolution dermoscopic surface micro-photographs. Utilizing a **MobileNetV2** backbone architecture optimized via transfer learning, the model processes skin images and calculates categorical probability profiles across 7 distinct clinical diagnostic classes.

---

##  Project Specifications & Scope
*   **Anatomical Region Focus:** Human Skin (Integumentary System)
*   **Imaging Modality Profile:** Dermoscopic Photographs (High-resolution surface microscopy)
*   **Problem Layout Category:** Multi-class Image Classification
*   **Target Conditions of Interest (7 Classes):**
    1.  *Actinic keratoses (akiec)* [Pre-cancerous]
    2.  *Basal cell carcinoma (bcc)* [Malignant]
    3.  *Benign keratosis-like lesions (bkl)* [Benign]
    4.  *Dermatofibroma (df)* [Benign]
    5.  *Melanoma (mel)* [Highly Malignant]
    6.  *Melanocytic nevi (nv)* [Benign Mole]
    7.  *Vascular lesions (vasc)* [Benign]
*   **Core Dataset Base:** HAM10000 Dataset (Shared, verified coursework source)
*   **Clinical Objective:** Build a low-latency screening pipeline capable of flagging aggressive surface pathologies to accelerate high-risk patient sorting without human bias.

 **CRITICAL MEDICAL DISCLAIMER:** This platform is an educational student coursework prototype. It is NOT an approved medical diagnostic tool or device. It is completely unsafe for autonomous medical evaluation, clinical decision-making, or real-world self-diagnosis.

---

## Repository File Inventory
*   `Cancer.ipynb`: Fully detailed notebook containing data acquisition configurations, strict patient-level splitting (`GroupShuffleSplit` anti-leakage control), model layer builds, training optimization paths, and final evaluation heatmaps.
*   `app.py`: Clean Streamlit diagnostic application containing cached resource model handling pipelines and plain-language interpretation summaries tailored for non-technical users.
*   `requirements.txt`: Comprehensive inventory of explicit working package version parameters required to isolate and run this system locally.
*   `skin_lesion_model.h5`: Compiled deep learning network file containing finalized trained weights.

---

##  Local Installation & Launch Manual
To execute the interactive Streamlit portal within a clean local environment workspace, execute these terminal commands step-by-step:

### 1. Project Directory Navigation
Open your system terminal window and change directory into this project repository folder:
```bash
cd path/to/your/Skin-Classifications
```

### 2. Environment Activation
Activate your workspace virtual environment wrapper:
```bash
source Medical...venv/bin/activate
# Windows Users use: venv\\Scripts\\activate
```

### 3. Dependencies Alignment Run
Install all programmatic requirements packages via the version checklist sheet:
```bash
pip install -r requirements.txt
```

### 4. Boot Up the Dashboard
Launch the interactive web portal platform locally:
```bash
streamlit run app.py
```
Once initialized, open a standard browser window tab and navigate to **`http://localhost:8501`** to interact with the system live.
