# DP GAN
This repository contains the code referred to in Time Series Synthesis Based on Differentially
Private Generative Adversarial Networks.

# Dependencies
This library requires Python 3.6. Dependencies have been specified in requirements.txt

To install Python dependencies:
> pip3 install -r requirements.txt

# PPG Dataset

# ROD Dataset
1. Download the occupancy_data.zip file from the UCI ML Database
https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+

2. Extract the occupancy_data.zip to the /ROD/ directory.

3. Run the following script to generate the dataset
    > python3 transform_room_sensor_time_series_with_occupancy_v1.py

4. Run the GAN with the following script:
    > python3 gan_v1.py

5. Set the folder_name variable in dp_mlp_v2.py to match the generated folder name

6. Run the DP Classifier with the following script:
    > python3 dp_mlp_v2.py 

# SAS Dataset
This GAN cannot be run due to its use of sensitive data. However, the code has been provided for examination