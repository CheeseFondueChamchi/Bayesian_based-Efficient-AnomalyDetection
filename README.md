- Resource-Efficient Anomaly Detection for 5G Basestation

This project introduces a solution to efficiently detect performance anomalies in radio units (RUs) within 5G networks, addressing the constraints of hardware resources. Utilizing Bayesian Adversarial Autoencoder (BAAE) and a Selective Re-Training (ST) algorithm based on uncertainty measures, this system offers an innovative approach to monitoring large-scale network environments.

-------------------------------------------------------------------------------------------------------------------------

- Solution Overview

To maintain reliable Quality of Service (QoS) in 5G networks, radio units must be monitored continuously for anomalies. The high computational load required for anomaly detection across a vast number of RUs presents a challenge. This project tackles that issue by implementing a selective retraining method that reduces unnecessary model updates, cutting computing costs while maintaining accuracy.

Anomaly Detection Model - BAAE
The BAAE model quantifies model uncertainty, allowing the system to update only when there is significant deviation in the monitored data.

<img width="953" alt="image" src="https://github.com/user-attachments/assets/7a4a9738-5e47-4da9-b8a2-0eafb92d28d6">

Performance Management (PM) Data
The PM data consists of key performance indicators that vary across environments and are critical to understanding equipment health. This data is the foundation for anomaly detection.

<img width="1017" alt="image" src="https://github.com/user-attachments/assets/fda3a0fb-0913-4dc2-a784-d0dced048a83">

STFT Analysis of PM Data
Short-Time Fourier Transform (STFT) is applied to PM data, transforming time-domain signals into frequency-domain representations, highlighting cyclical or irregular patterns indicative of network conditions.

<img width="1012" alt="image" src="https://github.com/user-attachments/assets/da759d0a-c037-4884-9a20-2be7f7c171fe">

Input Data Structure

The data used for training and monitoring contains multiple performance indicators per RU, collected periodically.

<img width="535" alt="image" src="https://github.com/user-attachments/assets/9cb95539-28d8-44eb-8fb5-53764d15c170">

-------------------------------------------------------------------------------------------------------------------------

- Algorithm

The Selective Re-Training (ST) algorithm calculates the uncertainty of each RU model, retraining only if significant outliers are detected, thus optimizing resource use.


<img width="523" alt="image" src="https://github.com/user-attachments/assets/1211de46-dd05-4e08-b6e5-15cc86f97723">

-------------------------------------------------------------------------------------------------------------------------

- Results

Our simulation demonstrates that the ST algorithm reduces retraining needs by approximately 40%, while also decreasing uncertainty outliers by up to 7.9%, showcasing its effectiveness in a resource-limited environment.

<img width="1020" alt="image" src="https://github.com/user-attachments/assets/1db9e043-e75d-49bf-b128-5c7e565d8797">


- Key Features

Efficient Resource Usage: Minimizes unnecessary model updates by selectively retraining based on uncertainty analysis.
Enhanced Reliability: Uses Bayesian methods to quantify uncertainty, ensuring high-confidence anomaly detection.
Scalability: Designed to handle a large volume of RUs across diverse network environments.
Future Work

Adaptive Real-Time Monitoring: Real-time adaptation to changing network conditions to further enhance QoS.
Extended Model Variants: Integrate other autoencoder-based models to improve anomaly detection across various data types.
