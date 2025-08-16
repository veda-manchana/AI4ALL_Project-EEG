# Emotional Recognition in Gaming Using EEG Signals

[![View on GitHub](https://img.shields.io/badge/View%20on-GitHub-black?logo=github)](https://github.com/veda-manchana/AI4ALL_Project-EEG)

## Problem Statement <!--- do not change this line -->

*Video games elicit complex emotional responses. Understanding these responses can enhance user experience, enable adaptive game design, and support mental health applications.*
1. *EEG-based emotion recognition provides a non-invasive method to measure brain activity in real-time and classify emotional states. Accurate detection of positive and    negative emotions allows:*
2. *Personalizing gaming experiences*
3. *Identifying stress or boredom triggers*
4. *Integrating emotion-adaptive mechanics*

This project applies machine learning to EEG (Electroencephalography) data from 27 participants to classify emotional states.  
The task was simplified into **binary classification** — grouping *Calm*, *Satisfaction*, and *Funny* as positive emotions, and *Boredom* and *Horrible* as negative emotions.

We tested **SVM**, **Logistic Regression**, and **Random Forest**, with Random Forest achieving the best performance.

## Key Results <!--- do not change this line -->

1. *Achieved **62.9%** accuracy in binary emotion classification using Random Forest*
2. *Random Forest outperformed SVM and Logistic Regression for EEG-based emotion recognition*
3. ***Feature importance** analysis showed distributed contributions across multiple brain regions rather than concentrated in a few channels*

## Methodologies <!--- do not change this line -->

1. *Collected EEG data from 27 participants playing 4 games (5 min each)*
2. *Preprocessed EEG signals: averaged channels, standardized features, and converted to binary labels*
3. *Split data into 80% training and 20% testing*
4. *Trained SVM, Logistic Regression, and Random Forest models*
5. *Evaluated models using accuracy and confusion matrices*
6. *Analyzed feature importance to understand brain-region contributions*

## Data Sources <!--- do not change this line -->

- **Source:** Cleaned EEG CSV files from 27 participants, 4 games each (5 minutes per game)  
- **Features:** Averaged EEG channels by brain region (frontal, temporal, parietal, occipital) plus mean and variance  
- **Labels:** Binary emotion labels — Positive (Calm, Funny, Satisfaction) and Negative (Boredom, Horrible)  

*Kaggle Datasets: [Link to Kaggle Dataset](https://www.kaggle.com/datasets/wajahat1064/emotion-recognition-using-eeg-and-computer-games/data)*

## Technologies Used <!--- do not change this line -->

- *Python*
- *pandas, numpy, scikit-learn*
- *Matplotlib, Seaborn (for visualizations)*

## Models Tested
- **SVM** — Moderate performance, struggled with high-dimensional space  
- **Logistic Regression** — Simple and interpretable, but underperformed  
- **Random Forest** — Best performance, handled non-linear relationships well  
- **Feature importance:** Spread across multiple brain regions rather than concentrated in a few channels

## Results
- **Accuracy:** 62.9%  
- **Confusion Matrix:**
[[275405 177988]
[113154 218837]]

## Authors <!--- do not change this line -->

*This project was completed in collaboration with:*
- *Veda Manchana ([manchana.veda@gmail.com](mailto:manchana.veda@gmail.com))*
- *Marzia Tahsin ([marziat1@umbc.edu](mailto:marziat1@umbc.edu))*
