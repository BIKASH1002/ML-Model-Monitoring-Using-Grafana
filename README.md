![banner](https://github.com/user-attachments/assets/fabd1fd3-2760-4068-8aef-84d2061757c9)

# Loan Prediction Model Monitoring with Prometheus and Grafana

<div align = "justify">

This project showcases a real-time monitoring solution for a loan prediction model using **Prometheus** and **Grafana**. It demonstrates the importance of observability by tracking key performance metrics such as accuracy, loss, precision, and recall. The setup ensures that model behavior is monitored seamlessly during training and production, helping detect issues early.

## Content

- [Overview](#overview)

- [Setup](#setup)

- [Procedure](#procedure)

- [Dashboard Key Metrics](#dashboard-key-metrics) 

- [Potential Improvements](#potential-improvements)

- [Conclusion](#conclusion)

## Overview

This project involves building and monitoring a loan prediction model using a RandomForestClassifier. With Prometheus collecting the model's performance metrics and Grafana visualizing these metrics on an interactive dashboard, the solution ensures real-time insights and helps detect performance bottlenecks early.

## Setup

- Visual Studio: for development

- Prometheus: to collect and store model metrics for monitoring

- Grafana: to visualize metric in real-time dashboard

- Docker: to containerize Prometheus and Grafana services

## Procedure

**1. Set up the monitoring stack**

- Ensure Docker is installed and run `docker-compose` to start Prometheus and Grafana.
   
- Access Prometheus at `http://localhost:9090` and confirm that the targets are active. Following screenshot shows the health of endpoint to verify if target is active or not:

![prometheus target](https://github.com/user-attachments/assets/91a48b79-54e1-4d66-8407-d47e8c63990a)

**2. Train the Loan Prediction Model**

- Run `train_model.py` to train the **RandomForestClassifier** model on the loan dataset.

- The model automatically exposes `metrics` such as accuracy and loss to Prometheus at `http://localhost:8000/metrics`. The logs can be seen as follows:

![prometheus metric endpoint](https://github.com/user-attachments/assets/9e5ea14b-4916-4f38-9c8a-ec0533c4be4d)

**3. Configure Grafana for visualization**

- Open Grafana and configure Prometheus as a data source.

- Create panels to visualize metrics like:

  - Accuracy and Loss over time.
    
  - Precision, Recall, and F1-Score.
    
  - Prediction Distribution.
  
**4. Monitor Scrape Duration**

- Add a Scrape Duration panel to ensure Prometheus collects data efficiently.
  
- Adjust the scraping intervals in `prometheus.yml` if you notice delays or dropped metrics.
  
**5. Evaluate Model Performance**

- Track how metrics change over time. For example:
  
  - A decreasing loss indicates the model is learning well.
    
  - High precision and recall reflect balanced predictions.
  
- Identify patterns or potential issues using the Prediction Distribution.

## Dashboard Key Metrics

- **Accuracy:** 0.95 – Reflects strong predictive performance.

- **Precision:** 0.91 – Ensures few false positives.

- **Recall:** 0.70 – Some false negatives but reasonable recall.

- **F1-Score:** 0.80 – A good balance between precision and recall.

- **Training Loss:** Tracks the error during training.

- **Prediction Distribution:** Shows the spread of predicted values.

- **Scrape Duration:** Ensures Prometheus collects data within the expected time.

Following is the Grafana dashboard representing the key metrics:

![grafana dashboard](https://github.com/user-attachments/assets/1425afea-98db-4fb4-a0b4-a6b1823a5a19)

## Potential Improvements

- **Adding Alerts:** Configure Grafana alerts to notify when metrics exceed thresholds (e.g., loss > 0.5).

- **Expanding Metrics:** Include additional metrics such as **AUC-ROC** or epoch-wise loss.

## Conclusion

This project demonstrates how to build, monitor and visualize a loan prediction model using a modern observability stack. With Prometheus collecting metrics and Grafana providing interactive dashboards, we gain real-time insights into our model’s behavior. This setup ensures that any performance bottlenecks or model degradation can be detected early, making it highly suitable for production environments.

</div>

## Credits

- **Kaggle:** for loan prediction dataset

## License

This project is licensed under the GPL-3.0 license.  
