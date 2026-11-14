# 50.020-WaDi
SUTD Final Group Project for Network Security on WaDi iTRUST dataset 

# Anomaly Detection for Attacks on Water Distribution System (WaDi) using Machine Learning Models (LSTM-AE, Isolation Forest)
    This project aims to build a system consisting of 
    1. a ML/DL model that detects anomalies in WaDi dataset 
    2. an LLM that inteprets anomaly outputs into operator language. 
   
            model -> LLM -> understandable context

### Unsupervised Machine Learning Models of Choice
1. #### Isolation Forest (Baseline)
    Lightweight ML model to detect anomalies or outliers in data

2. #### LSTM-AE (Long Short-Term Memory + AutoEncoder)
    DL (DeepLearning) model designed for sequential and time-series data


----- 

### Characteristics of dataset (WaDi.A2) 
1. 16 days of continuous operation: 14 under normal operation & 2 days with attack scenarios
2. Data from all the 123 sensors and actuators
3. Attack Scenarios: Derived from the attack models developed by iTRUST research team. The attack model considers the intent space of a CPS as an attack model. 15 attacks were launched during the 2 days.
4. Unstable readings from A1 dataset have been cleaned 
 

For more information/download dataset : https://www.sutd.edu.sg/itrust/itrust-labs/datasets/dataset-characteristics/wadi/

## To Run 
> Note: Please request to download dataset from iTRUST website, and put A2 dataset into this folder to run notebooks.  
> > File Name: [**WADI.A2_19 Nov 2019**]
> 
> Place 'WADI_14days_new.csv' and 'WADI_attackdataLABLE.csv' into 'dataset' folder. These two files have been .gitignore -d
