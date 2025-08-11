# Emotion Recognition in Gaming using EEG Signals

## Overview
This project applies machine learning to EEG (Electroencephalography) data from 27 participants to classify emotional states.  
The task was simplified into **binary classification** — grouping *Calm*, *Satisfaction*, and *Funny* as positive emotions, and *Boredom* and *Horrible* as negative emotions.

We tested **SVM**, **Logistic Regression**, and **Random Forest**, with Random Forest achieving the best performance.

---

## Dataset
- **Source:** Cleaned EEG CSV files from 27 participants, 4 games each (5 minutes per game)  
- **Features:** Averaged EEG channels by brain region (frontal, temporal, parietal, occipital) plus mean and variance  
- **Labels:** Binary emotion labels — Positive (Calm, Funny, Satisfaction) and Negative (Boredom, Horrible)  

---

## Preprocessing
1. Combined CSVs into a single DataFrame  
2. Standardized column names and units  
3. Added averaged brain region features  
4. Converted multi-class emotions into binary labels  
5. Split into **80% training** and **20% testing**

---

## Models Tested
- **SVM** — Moderate performance, struggled with high-dimensional space  
- **Logistic Regression** — Simple and interpretable, but underperformed  
- **Random Forest** — Best performance, handled non-linear relationships well  
  - **Feature importance:** Spread across multiple brain regions rather than concentrated in a few channels

---

## Results
- **Accuracy:** 62.9%  
- **Confusion Matrix:**
[[275405 177988]
[113154 218837]]
