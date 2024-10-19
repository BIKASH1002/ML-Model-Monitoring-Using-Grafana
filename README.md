# Loan Prediction Model Monitoring with Prometheus and Grafana

This project showcases a real-time monitoring solution for a loan prediction model using **Prometheus** and **Grafana**. It demonstrates the importance of observability by tracking key performance metrics such as accuracy, loss, precision, and recall. The setup ensures that model behavior is monitored seamlessly during training and production, helping detect issues early.

# Content

- Overview

- Setup

- Procedure

- Dashboard Key Metrics 

- Potential Improvements

- Conclusion

## Overview

This project leverages Prometheus to scrape and store metrics from a Python-based loan prediction model. Grafana visualizes these metrics in an interactive dashboard, giving real-time insights into model accuracy, loss, and system performance. This approach ensures that you can monitor your model’s behavior and detect performance degradation efficiently.

## Setup

- Visual Studio: for development

- Prometheus: to collect and store model metrics for monitoring

- Grafana: to visualize metric in real-time dashboard

- Docker: to containerize Prometheus and Grafana services

## Procedure

**1. Set up the monitoring stack**

- Ensure Docker is installed and run `docker-compose` to start Prometheus and Grafana.
   
- Access Prometheus at `http://localhost:9090` and confirm that the targets are active.

**2. Train the Loan Prediction Model**

- Run `train_model.py` to train the **RandomForestClassifier** model on the loan dataset.

- The model automatically exposes metrics such as accuracy and loss to Prometheus at `http://localhost:8000/metrics`.

**3. Visualize Metrics in Grafana**

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
