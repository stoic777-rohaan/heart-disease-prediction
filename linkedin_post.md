🫀 Built a Heart Disease Prediction model from scratch — sharing what I learned.

I worked with the Heart Failure Prediction dataset to build an end-to-end ML
pipeline:

📊 EDA — checked distributions of Age, Resting BP, Cholesterol, and Max Heart
Rate, and looked at how each related to heart disease risk.

🧹 Data cleaning — found that Cholesterol and Resting BP had invalid zero
values (physiologically impossible), and handled them by imputing column
means instead of just dropping rows.

🔠 Feature encoding — one-hot encoded categorical clinical features
(Chest Pain Type, Sex, ST Slope, etc.) for modeling.

🤖 Model comparison — trained and evaluated 5 classifiers (Logistic
Regression, Naive Bayes, Decision Tree, SVM, KNN) on accuracy and F1-score
to pick the best performer.

💻 Prototype app — wrote a Streamlit app that takes in patient details and
returns a risk prediction with probability.

For the polished, hosted version of the app, I used Emergent to help with
the UI/deployment — my focus for this project was the data science: cleaning
messy real-world health data, comparing models properly, and understanding
*why* one performed better than another, not just calling `.fit()`.

🔗 Live app: [link]
💻 Code + notebook: [GitHub link]

Would love feedback from anyone who's worked on clinical/health datasets —
curious how you handle imputation decisions like the zero-value issue I ran
into.

#MachineLearning #DataScience #Python #Streamlit #HealthTech #OpenToWork
