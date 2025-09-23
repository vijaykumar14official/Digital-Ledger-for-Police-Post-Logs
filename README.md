# 🛡️ Securecheck: Police Check Post Digital Ledger

## 📌 Overview
**Securecheck** is a Streamlit-powered interactive dashboard designed to digitize and analyze police check post records.  
It connects to a **MySQL database**, fetches real-time data, and provides **visual insights, metrics, and predictions**.  

The main goal of this project is to help law enforcement and traffic authorities track:
- Stop outcomes (arrests, warnings, tickets)
- Driver demographics (gender, race)
- Violations
- Time-based patterns
- Location-based insights
- Predictive insights for new police logs

---

## ✨ Features
- **Database Integration**: Connects to MySQL using SQLAlchemy.  
- **Dynamic Dashboard**:
  - 📊 Overall Metrics (Arrest rate, Search rate, Drug-related stops)  
  - 🚗 Vehicle Analysis (Top 5 frequently stopped vehicles)  
  - 👮 Demographics Insights (Stops by gender & race)  
  - ⚖️ Violation Analysis (Top violations, Trends, Outcomes)  
  - ⏳ Time & Duration Analysis (Stops by hour & duration distribution)  
  - 🌍 Location Insights (Top 5 stop locations)  
- **Interactive Forms**: Add new police logs directly through the dashboard.  
- **Prediction Module**: Suggests the most likely violation & stop outcome based on past records.  

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (Frontend & UI)
- **Pandas** (Data processing)
- **SQLAlchemy** (Database connection)
- **MySQL** (Backend database)
- **Plotly Express** (Interactive charts)

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
(https://github.com/vijaykumar14official/Digital-Ledger-for-Police-Post-Logs.git)
