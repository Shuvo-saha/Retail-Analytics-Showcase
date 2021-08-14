# Deep Dive into Retail Analytics
![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-Library-orange.svg)<br>
This **fully-fledged, interactive retail analytics showcase** dives into:
- The most important questions that big data and statistical learning can answer in the retail space
- The algorithms driving modern retail analytics
- How effective these algorithms can be in answering those questions 

You can view the web app [here](https://retail-analytics-showcase.herokuapp.com/)

---
## Data Size & Algorithms
- Recommendation System: Book ratings for approximately *5812 books* by *15,797 users*, cosine similarity based *Item-Item Collaborative Filtering*
- Market Basket Analysis: *7500* transactions at a grocery store, *Association Rules Analysis* using *Apriori*
- Churn Prediction: *3333 customers* and *17 features* per customer for a telecom operator, multiple classification models including:
  -  *Gradient Boosting Classifier* (81.56% F1-Score)
  -  *Decision Tree* (71.57% F1-Score)
  -  *Logistic Regression* (25.95% F1-Score)
  -  *Random Forest Classifier* (80.9% F1-Score)
  -  *Stacking Classifier* (85.25% F1-Score)
- Clustering: *8590* consumers and *17 different credit information* on each user of a financial institution, *K-Means Clustering* and *Principal Component Analysis*
- Customer Segmentation: Transactions covering *4380 Customers* and *4207 Products* for an e-commerce platform, *Recency, Frequency and Monetary Analysis*

---
## References
This website wouldn't have been possible without the helpful resources below:
- [Book Recommendations Dataset on Kaggle](https://www.kaggle.com/saurabhbagchi/books-dataset)
- [Churn Analysis Dataset on Kaggle](https://www.kaggle.com/sandipdatta/customer-churn-analysis)
- [Clustering Dataset on Kaggle](https://www.kaggle.com/ankits29/credit-card-customer-clustering-with-explanation)
- [RFM Analysis Dataset on Kaggle](https://www.kaggle.com/roshansharma/online-retail) 

For more information on Machine Learning, Statistics and Retail Analytics, check out these great resources:
- [Statquest for Statistics and ML](https://www.youtube.com/user/joshstarmer)
- [3Blue1Brown for Math](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
- [Sentdex for ML Coding Tutorials](https://www.youtube.com/user/sentdex)
- [Google Cloud Next'19 Keynote on Retail and AI](https://www.youtube.com/watch?v=pKEmQ1VMxsM)
- [DataRobot APAC Data Science's Presentation on Data Science in Retail](https://www.youtube.com/watch?v=PThNpfd3waE) 

This website was built using Python and the following libraries:
- [Streamlit](https://docs.streamlit.io/en/stable/index.html)
- [ELI5](https://eli5.readthedocs.io/en/latest/overview.html)
- [Scikit-Learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Pandas](https://pandas.pydata.org/) 

Deployment via [Heroku](https://www.heroku.com/)
