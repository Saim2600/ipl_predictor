# 🏏 IPL Win Predictor

A Machine Learning web application that predicts the **win probability of IPL teams in real-time** based on match conditions.

🌐 **Live App:** https://iplpredictor.streamlit.app/

---

## 🚀 Features

- 🔮 Real-time win probability prediction
- 📊 Uses match statistics (score, overs, wickets)
- 📈 Calculates CRR (Current Run Rate) & RRR (Required Run Rate)
- 🏟️ Considers match location (home ground advantage)
- ⚡ Interactive UI built with Streamlit

---

## 🧠 How It Works

1. Input match details:
   - Batting team
   - Bowling team
   - Score, overs, wickets
   - Target score
   - Match city

2. Model processes:
   - Remaining runs & balls
   - Current Run Rate (CRR)
   - Required Run Rate (RRR)

3. Outputs:
   - Winning probability for both teams

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Model:** Logistic Regression (Scikit-learn)  
- **Data Processing:** Pandas, NumPy  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure

ipl_predictor/
│── app.py              # Streamlit web app
│── retrain.py          # Model training script
│── pipe.pkl            # Trained ML model
│── team.pkl            # Team data
│── city.pkl            # City data
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

---

## ⚙️ Installation & Setup (Local)

```bash
git clone https://github.com/Saim2600/ipl_predictor.git
cd ipl_predictor
pip install -r requirements.txt
streamlit run app.py
