import re
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from eli5 import show_prediction, show_weights, explain_weights
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn import metrics
import os
from sklearn.model_selection import train_test_split
import eli5
from PIL import Image


@st.cache(allow_output_mutation=True)
def recommendation_engine():
    # Returns rating crosstab, merged_df_tr and coss_mat in that order
    merged_df_tr = pd.read_pickle(
        "Streamlit Data/Recommendations/merged_df_tr.bz2")
    rating_crosstab = pd.read_pickle(
        "Streamlit Data/Recommendations/rating_crosstab.bz2")
    coss_mat = np.load('Streamlit Data/Recommendations/coss_mat.npz')
    coss_mat = coss_mat["a"]
    return rating_crosstab, merged_df_tr, coss_mat


@st.cache(allow_output_mutation=True)
def association_analyzer():
    # Returns rating crosstab, merged_df_tr and coss_mat in that order
    rules = pd.read_pickle("Streamlit Data/Market Basket Analysis/rules.bz2")
    return rules


@st.cache(allow_output_mutation=True)
def churn_analyzer():
    # returns X,y,gbc scaler,churn_df_sample in that order
    gbc = joblib.load('Streamlit Data/Churn/gbc.joblib')
    dt = joblib.load('Streamlit Data/Churn/dt.joblib')
    lr = joblib.load('Streamlit Data/Churn/lr.joblib')
    rf = joblib.load('Streamlit Data/Churn//rf.joblib')
    stack = joblib.load('Streamlit Data/Churn//stack.joblib')
    X = np.load('Streamlit Data/Churn/X.npy')
    y = np.load('Streamlit Data/Churn/y.npy')

    with open('Streamlit Data/Churn/scaler', 'rb') as f:
        scaler = pickle.load(f)
    churn_df_sample = pd.read_pickle(
        "Streamlit Data/Churn/churn_df_sample.bz2")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_test, y_test, gbc, dt, lr, rf, stack, scaler, churn_df_sample


@st.cache(allow_output_mutation=True)
def clustering():
    # returns cluster_data, X_new in that order
    cluster_data_2 = pd.read_pickle(
        "Streamlit Data/Customer Segmentation/Clustering/cluster_data_2.bz2")
    cluster_data_3 = pd.read_pickle(
        "Streamlit Data/Customer Segmentation/Clustering/cluster_data_3.bz2")
    cluster_data_4 = pd.read_pickle(
        "Streamlit Data/Customer Segmentation/Clustering/cluster_data_4.bz2")
    cluster_data_5 = pd.read_pickle(
        "Streamlit Data/Customer Segmentation/Clustering/cluster_data_5.bz2")
    with open('Streamlit Data/Customer Segmentation/Clustering/X_new', 'rb') as X_new_file:
        X_new = pickle.load(X_new_file)
    return cluster_data_2, cluster_data_3, cluster_data_4, cluster_data_5, X_new

# this is needed for the recommendation system


@st.cache(allow_output_mutation=True)
def rfm_loader():
    rfm = pd.read_pickle("Streamlit Data/Customer Segmentation/RFM/rfm.bz2")
    return rfm


@st.cache
def rand_gen(x, y):
    return np.random.randint(x, y)


def model_formatter(x):
    word = re.sub(r"(\w)([A-Z])", r"\1 \2", type(x).__name__)
    return word


# PAGE BEGINNER
st.set_page_config(
    page_title="Impactful Retail Analytics - Powered by Intelligent Machines",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

main1, main2, main3 = st.beta_columns([1, 2, 1])
with main2:
    st.image(Image.open("Streamlit Data/Images/color_logo.png"), width=150)
    st.markdown("# Impactful Retail Analytics")
    st.markdown("""
                The retail world generates **massive amounts of data** from consumers and the interactions they have. 
                All that data can lead to **better customer targeting, higher 
                customer satisfaction and in turn, less customer churn**. With large datasets, however, we also need to ask the right questions.\n 
                To better understand how data can help retail businesses, we've created this **fully-fledged, interactive retail analytics showcase**. 
                The website will guide you through scenarios for a common retail businesses, analyze their existing data and answer important 
                business questions.\n 
                You can select any question from the box below to continue.
                """)
    # RECOMMENDATIONS

    question = st.selectbox(label="", options=['Pick a question',
                                               'What products do we recommend to our users?', 'What products increase the chance of buying other products?',
                                               'What customers are we likely to lose soon?', 'How can we segment our customers?', 'Who are our most important customers?', 'References and Further Reading'])

    if question == "Pick a question":
        st.markdown(
            "## **What do industry leaders say?**")
        st.write('''The video below goes through Boston Consultancy Group's thoughts on how analytics will change the retail
                 landscape.''')
        st.video("https://www.youtube.com/watch?v=AOV6bqpCV6k&t=6s")

    if question == 'What products do we recommend to our users?':
        st.markdown('# What products do we recommend to our users?')

        # this is needed for the recommender
        random_no = rand_gen(0, 7100)
        # Load the data
        # make a random selection of 3 books to pick from
        # take the book title from the user and return the corresponding ISBN
        st.write("A bookseller, just like Barnes & Nobles, wants to recommend their users new books based on their favorite books. All they have is a very rich dataset of their many users and their book preferences.")

        st.markdown("The dataset contains approximately *5812 books* and *15,797 users*. Below, we have 10 rows from the dataset. The ratings are on a scale of 0-10 with 0 being the lowest rating.  How can they make intelligent book recommendations?")
        st.image(Image.open("Streamlit Data/Images/book.png"), width=50)

        rating_crosstab, merged_df_tr, coss_mat = recommendation_engine()
        st.write(merged_df_tr[["Book", "User-ID", "Book-Rating"]]
                 [20:30].assign(hack='').set_index('hack'))

        st.markdown("## **Recommendation Systems are the answer!**")
        st.markdown('''Using *Item-Item Collaborative Filtering*, we can understand the similarities between books. 
                    Item-item collaborative filtering is based on the similarity between items calculated using people's ratings of
                    those items. Item-item collaborative filtering was invented and used by *Amazon* in 1998. To learn more, go to And here it is in action:''')
        with st.beta_expander("Learn more about Item-Item Collaborative Filtering"):
            st.markdown('''
                        Suppose that a customer, Mark, wants to purchase a book. We first look at all the books Mark has read - Harry Potter and Oliver Twist. Mark has rated both these books
                        highly. Three other users, Mathew and his brothers, also love these books and they also love Eragon. Therefore, these users rated Harry Potter, Oliver Twist and Eragon very similarly.
                        Two other users, Joan and Samantha, hate all three books. This also reinforces the fact that the books are similar. Therefore, we can recommend Eragon to Mark.\n
                        The exact process and math is a bit complicated but is explained clearly in this [article](https://towardsdatascience.com/comprehensive-guide-on-item-based-recommendation-systems-d67e40e2b75d).
                        ''')
        st.markdown(''' 
                    We've gone ahead and trained an Item-Item Collaborative Filtering model using the full dataset. The model will take the book name and then using the magic of [similarity measures](https://en.wikipedia.org/wiki/Similarity_measure#:~:text=In%20statistics%20and%20related%20fields,the%20similarity%20between%20two%20objects.),
                    (we've used *Cosine Similarity* for our analysis) it recommends the closest 3 books that you might like.  You can even search up the book names and check if we've got it right 😉
                    ''')
        with st.beta_expander("Learn more about Cosine Similarity"):
            st.markdown('''
                        A simple similarity measure is Euclidean distance or , the length of the shortest possible path between two points.\n
                        Cosine similarity is similar but it does not take size into account. Mathematically, it measures the cosine of the angle 
                        between two vectors projected in a multi-dimensional space. This property is very useful when dealing with multidimensional
                        spaces such as our user ratings (imagine every user is an axis). Thus Cosine Similarity better classifies similar books compared to Euclidean Distance.
                        To learn more, go to this [article](https://www.machinelearningplus.com/nlp/cosine-similarity/).
                        ''')
            st.image(Image.open(
                "Streamlit Data/Images/cosine-similarity.png"), width=500)
        st.markdown("## **Select a book to generate top 3 recommendations.**")
        search = st.selectbox(label="",
                              options=list(merged_df_tr["Book"][random_no:random_no+100]), help="100 books are randomly selected for the selection pool")
        submit_rec = st.button("Submit")
        if submit_rec:
            search = merged_df_tr.loc[merged_df_tr["Book"]
                                      == search, "ISBN"].values[0]
            # with the ISBN found, we can now continue with the original method
            col_idx = rating_crosstab.columns.get_loc(search)
            corr_specific = coss_mat[col_idx]
            top_items = pd.DataFrame({'corr_specific': corr_specific, 'ID': rating_crosstab.columns})\
                .sort_values('corr_specific', ascending=False)\
                .head(10)
            top_items = top_items.merge(merged_df_tr[[
                                        "Book-Title", "Book-Author", "Book", "ISBN"]].drop_duplicates(), left_on="ID", right_on="ISBN", how="left")
            common_items = set(merged_df_tr.loc[merged_df_tr["Book-Title"] == top_items.iloc[0, 2], "User-ID"].sort_values(
            )).intersection(set(merged_df_tr.loc[merged_df_tr["Book-Title"] == top_items.iloc[1, 2], "User-ID"].sort_values()))
            similar = ", ".join(top_items["Book"][1:4].values.tolist())
            # st.write("Books similar to", top_items["Book"][0], "are:")
            st.write(top_items["Book"][1:4])

    # MARKET BASKET ANALYSIS
    if question == 'What products increase the chance of buying other products?':
        rules = association_analyzer()
        st.markdown(
            '# What products increase the chance of buying other products?')

        st.write('''
                A grocery store chain, just like Walmart, wants to increase basket size of its consumers. 
                If the retailer were to know hidden associations between products, he/she could put related 
                products in close proximity and give offers (a small nudge) for one of the related products.\n
                Also, note that the problem isn't the same as recommending products. Instead, this time,
                the store is trying to find out products that *go well together*.\n  
                The grocery store has a large dataset of transactions of their current users. Every row in the dataset, shown below,
                contains the basket of goods that was bought by users. There are *7500 such transactions*. Can they find out
                products that go well with other products using data alone?
                ''')
        st.image(Image.open("Streamlit Data/Images/grocery.png"), width=50)

        st.write(pd.DataFrame({"Item 1": ['burgers', 'meatballs', 'eggs'],
                               "Item 2": ['chutney', 'salsa', 'ketchup'],
                               "Item 3": ['turkey', 'avocado', 'peaches']}).assign(hack='').set_index('hack'))
        st.markdown(
            "## **Market Basket Analysis can uncover hidden product associations.**")
        st.markdown('''
                    Using Market Basket Analysis, we can uncover items that are frequently brought together by the customer 
                    while also accounting for product popularity (so popular products won't keep being in the top).\n
                    To understand the analysis, we need to understand a few terms. 
                    - *An association rule* is a rule that describes how the first product (antecedent) behaves
                    in regard to the second product (consequent). E.g. if a customer buys Coca Cola, he'll also buy Hot Dog Buns.
                    - *Support* is the popularity of an item. This is a very misleading statistic as we need to take care
                    to penalize popular products. For example, since soap is very popular, it'll appear in every basket that
                    Baby Powder appears in but this doesn't mean that soap itself increases the likelihood of buying Baby Powder.
                    On the other hand, we should take care to also not let niche products and one off transactions
                    skew our analysis.
                    - *Confidence* is the likelihood of the association rule being true. A higher value is better.
                    - *Lift* is the increase in likelihood of a consumer buying the second product, given he bough the first product
                    e.g. a lift score of 3 for diapers -> milk means that if the customer had a 30% likelihood of buying milk on any
                    day, that likelihood is now 90% (30*3).\n  
                    ''')
        # _, col_im, _ = st.beta_columns([1, 1, 1])
        st.image("Streamlit Data/Images/equations-mba.png",
                 caption="Support, Confidence and Lift Formulas", width=500)

        st.markdown('''
                    ---
                    We also need an algorithm to prune the large number of possible combinations. *Apriori* is one such popular algorithm.
                    In simple terms, we know that unpopular items will also have unpopular combinations. Therefore, using a cutoff support or popularity
                    value, we can drastically reduce combinations of items, and thus, computation time. And that's all that Apriori is.\n 
                    Therefore, to understand product relationships, we can look at the **Lift** score, while talking into account,
                    the confidence (to measure reliability) and popularity.
                    ''')

        # col1, col2, col3 = st.beta_columns(3)
        st.markdown(
            "## **Use the cutoff values below to generate product associations.**")
        lift = st.slider(
            label="Choose Lift cutoff value", min_value=1.0, max_value=4.0, value=2.0)
        conf = st.slider(label="Choose Confidence cutoff value", min_value=0.0,
                         max_value=1.0, value=0.4)
        support = st.slider(label="Choose Support cutoff value", min_value=0.0,
                            max_value=0.06, value=0.01)

        chosen = rules[(rules['lift'] >= lift) & (rules['confidence'] >= conf) & (
            rules['support'] >= support)].sort_values("lift", ascending=False).head(20)

        submit_mba = st.button("Submit")
        if submit_mba:
            chosen_expander = st.beta_expander(
                label='See generated association rules table and learn more about association rules')
            with chosen_expander:
                st.write(chosen.assign(hack='').set_index('hack'))
                st.markdown('''
                            You'll notice that the table above also has a conviction value. To learn more about association rules, Apriori (and conviction) go this [link](https://en.wikipedia.org/wiki/Association_rule_learning)
                            and this [link](https://searchbusinessanalytics.techtarget.com/definition/association-rules-in-data-mining).
                            ''')

            def likelihood(x):
                if x.iloc[0, 6] > 2:
                    criteria = "extremely likely"
                elif x.iloc[0, 6] > 1.5:
                    criteria = "highly likely"
                elif x.iloc[0, 6] > 1:
                    criteria = "ikely"
                return criteria

            if chosen.empty:
                st.write("No strong associations found")
            elif (chosen.iloc[0, 6] <= 1):
                st.write("No strong associations found")
            else:
                likelihood1 = likelihood(chosen)
                st.markdown(
                    f"If a consumer buys **{chosen.iloc[0, 0]}** then he/she is also **{likelihood1}** to buy **{chosen.iloc[0, 1]}**.")

    # CHURN ANALYSIS
    if question == 'What customers are we likely to lose soon?':
        # load in the data
        X_test, y_test, gbc, dt, lr, rf, stack, scaler, churn_df_sample = churn_analyzer()

        st.markdown('# What customers are we likely to lose soon?')
        st.markdown('''
                A telecom service, like AT&T, wants to know which customers are likely to switch to their competitor.
                The company has a dataset of its *3333 customers* and *17 features* per customer. The company also knows which 
                customers have shifted to competitor services.The dataset sample is shown below:\n
                ''')
        with st.beta_expander("See feature descriptions"):
            st.markdown('''
                        - *Account length*, which is  how many days the customer has used the operator's services.
                - *International plan*, where 0 indicates the customer is not on the plan and 1 indicates the opposite.
                - *Voice mail plan*, where 0 indicates voice mail is not activated and 1 indicates the customer has voice mail activated.
                - *Number vmail messages* indicates the number of voice mail messages the customer has received in the past.
                - Columns with *calls* indicates number of calls during that time period.
                - Columns with *minutes* indicates total minutes spent talking during that time period.
                - Columns with *charge* indicate the amount of money charged to the customer during that time period.
                - *Customer service calls* is the number of calls the customer has made to customer service.\n
                Note that international calls have separate call, charge and minutes data.
                        ''')

        st.write(churn_df_sample.assign(hack='').set_index('hack'))
        st.write(
            "Can it use its past customer data to predict churn of new customers?")

        st.markdown('''
            ## **Using a classification model, we can find out which traits lead to churn.** 
            A *classification model* is a fancy way of saying that the model tries to predict
            classes rather than output a numeric value. In our case, we have two classes - churned
            and not churned.\n
            There are many classification techniques from simple linear methods like *logistic
            regression* to more complex methods like *neural networks*. We've picked 5 common models for
            this problem set. Pick any model below and we'll show you how it works, and how it performs.
            
            ''')
        st.markdown("---")

        # This is PFI, tkae  the text, format as df then sort,rename col and add color style
        # selection for different models
        model = st.selectbox(label="Pick classification algorithm", options=[
            gbc, dt, lr, rf, stack], format_func=model_formatter)
        y_pred = model.predict(X_test)
        if model == gbc:

            st.markdown('''
                    A *Gradient Boosting Classifier* is a model that uses a series of weak learners to make predictions.
                    It is an *ensemble* technique since it uses multiple models.\n
                    The algorithm/model minimizes a loss function e.g.
                    regression uses the squared error for its loss function. The weak learners are usually shallow
                    decision trees (select Decision
                    Tree Classifier for a thorough explanation). Then the trees are added one at a time and
                    loss is reduced gradually using a *gradient descent procedure*. More trees are added to
                    reduce loss until we've improved the final output or we've reached a maximum number of trees.
                    For more on Gradient Boosting, go over [this article](https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab). 
                    ''')

            st.image(Image.open("Streamlit Data/Images/boosting.png"),
                     caption="Source: Corporate Finance Institute", width=500)
            st.markdown('''
                    ---
                    The model predicts that customers with *high number of customer service calls, who mostly talk during 
                    the day, and have a high day charge* are likely to churn. The other features are listed in order of importance below.
                    Note that the values are not important for our analysis but the rankings are.
                    ''')
        elif model == dt:
            st.markdown('''
            A *Decision Tree Classifier* breaks down a dataset into smaller and smaller subsets based on certain cutoffs. Thus, the
            number of examples get smaller every division. Once the tree reaches a certain *depth* (or number of splits) or it
            has leaves that are *pure* (all examples belong to one class), it stops dividing.
            For more on Decision Trees, go to [this article](https://towardsdatascience.com/decision-tree-in-machine-learning-e380942a4c96). 
            ''')

            st.image(Image.open("Streamlit Data/Images/rain.png"),
                     caption="Rain Forecasting Decision Tree, Source: Prince Yadav, Towards Data Science", width=500)
            st.markdown('''The model predicts that customers with *high number of customer service calls, who mostly talk during 
                    the day, and have a high day charge* are likely to churn. The other features are listed in order of importance below.
                    Note that the values are not important for our analysis but the rankings are.
                    ''')
        elif model == lr:
            st.markdown('''
            *Logistic Regression* is similar to linear regression but instead
            it uses a function that squeezes the output to a scale of 0 to 1. 
            Using a cutoff value (usually 0.5),
            it makes binary predictions. Logistic Regression is explained
            in more detail in this excellent video by Josh Starmer:
            ''')
            st.video("https://www.youtube.com/watch?v=yIYKR4sgzI8")
            st.write('''
                    The model predicts that customers with *high number of customer service calls, an international plan,
                    and a high day charge* are likely to churn. The other features are listed in order of importance below.
                    Note that the values are not important for our analysis but the rankings are.
                    ''')
        elif model == rf:
            st.markdown('''
            *Random Forest Classifiers* train multiple decision trees (Pick Decision Tree Classifier for a thorough explanation)
            on subsets of the observations and features and then combine them for a more accurate predictions. Even though
            they're not complicated to understand, they're often very accurate and robust.\n
            The model predicts that customers with *high number of customer service calls, who mostly talk during 
                    the day, and have a high day charge* are likely to churn. The other features are listed in order of importance below.
                    Note that the values are not important for our analysis but the rankings are.
            ''')
        elif model == stack:
            st.markdown('''
            *Stacking, Bagging and Boosting* are all *ensemble* algorithms. 
            - *Bagging* uses
            homogeneous weak learners, trains them independently in parallel and then
            combines them using a deterministic method like averaging.
            - *Boosting* also uses homogeneous weak learners, but instead of combining
            them, it trains them sequentially and then uses a deterministic method like averaging.
            - *Stacking* uses heterogeneous or different types of weak learners, trains
            them in parallel and combines them by using another model (like
            logistic regression) to output a prediction.
            Note that the feature importances for stacking models are not shown below as 
            it is difficult to calculate and may not be accurate. However, the accuracy usually
            improves when combining multiple models (and has improved in our case as well). 
            ''')

        # if stacking classifier this doesn't work so make sure to use if function here
        if model != stack:
            text_fi = explain_weights(model, targets=[False, True], feature_names=['account length', 'international plan', 'voice mail plan',
                                                                                   'number vmail messages', 'total day minutes', 'total day calls',
                                                                                   'total day charge', 'total eve minutes', 'total eve calls',
                                                                                   'total eve charge', 'total night minutes', 'total night calls',
                                                                                   'total night charge', 'total intl minutes', 'total intl calls',
                                                                                   'total intl charge', 'customer service calls'], top=len(['account length', 'international plan', 'voice mail plan',
                                                                                                                                            'number vmail messages', 'total day minutes', 'total day calls',
                                                                                                                                            'total day charge', 'total eve minutes', 'total eve calls',
                                                                                                                                            'total eve charge', 'total night minutes', 'total night calls',
                                                                                                                                            'total night charge', 'total intl minutes', 'total intl calls',
                                                                                                                                            'total intl charge', 'customer service calls'])+1)
            text_fi = eli5.formatters.as_dataframe.format_as_dataframe(text_fi)
            pfi_table = text_fi[['feature', 'weight']].sort_values(
                "weight", ascending=False).rename(columns={'feature': "Features", "weight": "Weights"}).style.background_gradient(axis=0)
            with st.beta_expander("See explanation of feature importance"):
                st.markdown('''
                            Most machine learning models are *black-box* models. This means that they're created
                            directly from data and even the people who designed them, cannot understand how
                            the variables are being combined to make predictions. Methods to explain these models
                            are usually in two forms: local and global explainers.\n 
                            Local explainers allow us to
                            explain a specific prediction e.g. *LIME (Local Interpretable Model-agnostic Explanations)*
                            creates a white-box or explainable model that works well in the area close to the concerned
                            prediction example.\n
                            Global explainers are used to explain the whole model and how every feature impacts it on a
                            general level e.g. *Permutation Feature Importance (PFI)* looks at how much the score (R-Squared, F-1 score, etc)
                            decreases when a feature is not available. The values below are generated using [PFI and the ELI5 libary](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html).
                            ''')
            st.write(pfi_table)
        # print a confusion matrix using test data
        y_pred = model.predict(X_test)

        # Plot conf matrix and f1 score in an expander
        with st.beta_expander(
                label='See classification accuracy and other metrics for the selected model'):
            # col_i, col_ii = st.beta_columns([2, 5])
            conf_fig = plt.figure(figsize=(5, 4))
            conf_matrix = metrics.confusion_matrix(y_pred, y_test)
            sns.heatmap(conf_matrix, annot=True, xticklabels=[
                        'Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.title("Confusion Matrix")
            buf = BytesIO()
            conf_fig.savefig(buf, format="png")
            st.image(buf, width=500)
            # F1 score
            st.markdown(
                f'''
                To understand accuracy of our model, we can make a *confusion matrix* which plots the *predicted*
                values against  the *actual/true* values. This data is different from our training data and is
                usually called the *test/validation* data. Low values in the top right and bottom left are better. \n
                Accuracy is not [a good metric for classification with unbalanced classes](https://medium.com/analytics-vidhya/accuracy-vs-f1-score-6258237beca2#:~:text=Accuracy%20is%20used%20when%20the,and%20False%20Positives%20are%20crucial&text=In%20most%20real%2Dlife%20classification,to%20evaluate%20our%20model%20on.). Instead, the *F1 Score* is used
                which is a weighted average of the precision and recall. The *precision* is the number of positive class 
                predictions that actually belong to the positive class while the *recall* quantifies the number of positive class 
                predictions made out of all positive examples in the dataset.\n
                F1 score for **{model_formatter(model)}** is **{round((metrics.f1_score(y_test, y_pred))*100,2)}%**.
                Recall score is **{round((metrics.recall_score(y_test, y_pred))*100,2)}%** and 
                Precision score is **{round((metrics.precision_score(y_test, y_pred))*100,2)}%**
                ''')
            st.write(
                f'')

        st.write('''
                 ## **Try out the churn model!**
                Our model can be tested against an imaginary consumer with your selected features. 
                Note that other features have been preset for convenience.
                ''')
        st.info('''Be aware that none of the models have 100% accuracy (especially Logistic Regression) and thus may output
                wrong predictions.''')
        # make 6 different columns
        # col4, col5, col6, col7, col8, col9 = st.beta_columns(6)

        acc_length = st.number_input("Account length", min_value=0, step=1)
        int_plan = st.number_input(
            "International Plan", min_value=0, max_value=1, step=1)
        vml_plan = st.number_input(
            "Voice Mail Plan", min_value=0, max_value=1, step=1)
        num_vm = st.number_input(
            "Number of Voice Mail Messages", min_value=0, step=1)
        total_day = st.number_input("Total Day Minutes", min_value=0)
        customer_service = st.number_input(
            "Customer Service Calls", min_value=0, step=1)

        submit_churn = st.button("Submit")
        if submit_churn:
            test_arr = np.array([[acc_length, int_plan, vml_plan, num_vm, total_day, 137, 21.95, 228.5, 83, 19.42, 20, 11, 9.4,
                                  12.7, 6, 3.43, customer_service]])
            if gbc.predict(scaler.transform(test_arr))[0] == False:
                st.markdown(
                    "The model predicts the customer is **unlikely** to switch to the competitor (given the dataset)")
            elif gbc.predict(scaler.transform(test_arr))[0] == True:
                st.markdown(
                    "The model predicts the customer is **likely** to switch to the competitor (given the dataset)")

    # CUSTOMER SEGMENTATION - CLUSTERING
    if question == 'How can we segment our customers?':
        st.markdown('# How can we segment our customers?')
        st.markdown('''
                    A superstore, like CostCo, wants to collaborate with a banking institution
                    like Bank of America to provide attractive offers to their customers. The
                    banking institution wants to reward it's top customers as well. Therefore,
                    both institutions have a mutual interest.\n
                    The banking institution has a dataset consisting of the credit information
                    of the mutual consumers of the two companies. The dataset has *8590* consumers and
                    *17 different credit information* on each user. Can the institution segment its 
                    customers based on the data provided?
                    ''')
        with st.beta_expander("See feature descriptions"):
            st.markdown('''
                    
                    - BALANCE : Balance amount left in their account to make purchases
                    - BALANCEFREQUENCY : How frequently the balance is updated; score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
                    - PURCHASES : Value of purchases made
                    - ONEOFFPURCHASES : Maximum purchase amount done in one go
                    - INSTALLMENTSPURCHASES : Value of purchase done in installment
                    - CASHADVANCE : Cash in advance given by the user
                    - PURCHASESFREQUENCY : How frequently the purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
                    - ONEOFFPURCHASESFREQUENCY : How frequently large one-off purchases are happening (1 = frequently purchased, 0 = not frequently purchased)
                    - PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
                    - CASHADVANCEFREQUENCY : How frequently the cash in advance is being paid
                    - CASHADVANCETRX : Number of transactions made with "cash in advance"
                    - PURCHASESTRX : Number of purchase transactions made
                    - CREDITLIMIT : Limit of credit card for user
                    - PAYMENTS : Amount of payment done by user
                    - MINIMUM_PAYMENTS : Minimum amount of payments made by user
                    - PRCFULLPAYMENT : Percent of full payment paid by user
                    - TENURE : Tenure of credit card service
                    ''')

        cluster_data_2, cluster_data_3, cluster_data_4, cluster_data_5, X_new = clustering()
        st.write(cluster_data_2.drop("cluster_id", axis=1).head())

        st.markdown('''
                ## **K-Means Clustering can be used to cluster large, multidimensional datasets**
                This problem can be treated as a *clustering problem* - a problem where we try to
                group similar data points to make heterogeneous clusters. One can imagine a 2d graph
                where the points are separated into two clearly defined clusters.\n
                In real life, the problem is more complicated. The data is usually multidimensional
                and thus cannot be separated based on a simple visual analysis, as in 3 dimensions or less.
                The data is usually also very large and may not be perfectly separable into distinct clusters.
                Additionally, clustering is an unsupervised problem (no information is given to the model
                to let it know if it has gotten the answer right), is entirely based on the distribution
                of the data and is inaccurate in most cases.\n
                ---
                While not the most efficient or easy to use clustering algorithm, *K-Means* is still very effective.
                The general idea is:
                - Select k clusters based on domain knowledge e.g. in our case, the banking institution says that
                most customers are usually in 3 segments - low, medium and high risk.
                - Randomly initialize k centroids, and assign each data point to the closest centroid. Thus, initial
                random k clusters are formed.
                - Select the whole cluster and find its average point. This is the new centroid.
                - Assign the points to the new centroid. If points have moved from one cluster to another,
                repeat the above process.
                We usually stop after a set number of iterations or until the points do not move to another 
                cluster. See the video below for a visual example.
                ''')
        st.video("https://www.youtube.com/watch?v=5I3Ei69I40s")
        st.markdown('''
                    ## **Use K-Means Clustering to find similar consumers.**
                    The cluster number can be selected below and the results of the clustering can be visualized using the
                    *Principal Components* graph and scatter graph. You'll notice that 2 to 3 clusters usually work best.
                    ''')
        clusters = st.select_slider("How many clusters?", [2, 3, 4, 5])
        if clusters == 2:
            cluster_data = cluster_data_2
        elif clusters == 3:
            cluster_data = cluster_data_3
        elif clusters == 4:
            cluster_data = cluster_data_4
        elif clusters == 5:
            cluster_data = cluster_data_5
        st.markdown('''
                    *Principal Component Analysis* or PCA is a dimensionality reduction technique 
                    that transofrms a large set of features into less features that still 
                    contains most of the information in the large set. While the statistics is a 
                    bit complicated, it is fairly easy to [calculate](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
                    and is explained in more detail [here](https://www.youtube.com/watch?v=FgakZw6K1QQ).\n
                    We've used *2 Principal Components* that allow us to visualize all 17 features on 2 dimensions
                    while retaining *65.31% variance* of the underlying data (graph on bottom left). You can also view the formed clusters on any two of your preferred features
                    using the options below.
                    ''')
        # st, cold = st.beta_columns(2)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax = sns.scatterplot(x=X_new[:, 0], y=X_new[:, 1],
                             hue=cluster_data.cluster_id.values, palette="pastel").set_title('Principal Components')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.info("The different colors represent different clusters")

        st.image(buf)
        x = st.selectbox(label="Select X axis",
                         options=cluster_data.drop("cluster_id", axis=1).columns)
        y = st.selectbox(label="Select Y axis", options=cluster_data.drop(
            "cluster_id", axis=1).columns.sort_values(ascending=False))

        st.markdown('''
                    
                    ''')

        fig2, ax2 = plt.subplots(figsize=(7, 6))
        ax2 = sns.scatterplot(data=cluster_data, x=x, y=y,
                              hue=cluster_data.cluster_id.values, palette="pastel")
        buf2 = BytesIO()
        fig2.savefig(buf2, format="png")
        st.image(buf2)
        st.write('''
                 ---
                What do these clusters represent? Let's take a look at the Credit Limit vs Purchases graph for
                clustering with 2 and 3 clusters.\n
                For 2 clusters, the model has clustered customers with low credit card usage in
                    one cluster and higher usage in another cluster.
                    For 3 clusters, the model has put customers with unusually high amounts of purchases
                    and credit limits in one cluster and customers with very low amounts in another.
                ''')

    # CUSTOMER SEGMENTATION - RFM
    if question == 'Who are our most important customers?':
        rfm = rfm_loader()
        st.markdown('# Who are our most important customers?')
        st.markdown('''
                An e-commerce firm, like Flipkart, wants to find out who their most valuable customers are, which customers
                they're likely to lose and which customers require a little nudge to purchase more. They have a
                rich dataset of transactions but no customer information (other than customer ID). A small sample is shown below:
                ''')
        st.write(pd.DataFrame({"InvoiceNo": [536365, 536362, 536323], "CustomerID": [2132, 2134, 1123], "StockCode": [
            123, 145, 121], "Quantity": [2, 6, 2], "UnitPrice": [51.6, 212.5, 12.4], "InvoiceDate": ["1/12/2010", "6/12/2010", "7/12/2010"]}))
        st.write(
            "Can the e-commerce firm categorize their customers and find out the most important segments?")
        st.markdown('''
                ## **RFM Analysis can help find interesting customer segments.**
                RFM stands for **Recency, Frequency and Monetary** value analysis. What do these mean?
                - *Recency* is the freshness of a customer. For example, a customer who bought an item in the last day would
                have a very high recency score. For our analysis, we'll look at how many days ago a customer made a purchase.
                - *Frequency* is how frequently a customer buys from our store or visits our store. For our analysis, we'll look at how many 
                times a customer made a purchase.
                - *Monetary* is the purchasing power of the consumer. For our analysis, we'll look at total lifetime spending.\n
                ---
                Note that actual measures of the metrics vary from business to business and the measures we're using are only
                applicable to the business we're dealing with. Let's take a look at the results (a small sample, at least) of our analysis below:
                
                ''')
        st.write(rfm.rename(columns={"rankR": "Rank Recency", "rankF": "Rank Frequency", "rankM": "Rank Monetary", "Customer_Segment": "Customer Segment"}).dropna(
        ).drop_duplicates("Customer Segment").set_index('Name'))
        st.markdown('''
                    From the table above, we can see the 3 metrics but also some other columns we haven't looked at. The *Rank*
                    columns are simple rankings based on the measures. The *Customer Segment* column is the ultimate result of the
                    analysis. Some interesting segments are:\n
                    - *Champions* are your best customers, those who buy often, are engaged, and spend a lot. They are the most
                    valuable customers, so we advice you always keep an eye on them!
                    - *Potential Loyalists* are recent customers who spend a lot. We want to increase their purchase frequency
                    so loyalty programs can help here.
                    - *At Risk* are customers who purchase often and spent a lot but haven't purchased recently. Get them back
                    with some discounts and offers!
                    
                    ''')
        with st.beta_expander("Learn more about RFM analysis"):
            st.markdown('''
                        To learn more about RFM analysis and the underlying methodologies, go to this [article](https://clevertap.com/blog/rfm-analysis/).
                        [This analysis](https://www.kaggle.com/blewitts/ecommerce-rfm-analysis) on Kaggle also goes over the other customer
                        segments and explains the metrics (it was also an inspiration for this analysis ❤️).
                        ''')
    # REFERENCES
    if question == "References and Further Reading":
        st.markdown('''
                    This website wouldn't have been possible without the helpful resources below:
                    - [Book Recommendations Dataset on Kaggle](https://www.kaggle.com/saurabhbagchi/books-dataset)
                    - [Churn Analysis Dataset on Kaggle](https://www.kaggle.com/sandipdatta/customer-churn-analysis)
                    - [Clustering Dataset on Kaggle](https://www.kaggle.com/ankits29/credit-card-customer-clustering-with-explanation)
                    - [RFM Analysis Dataset on Kaggle](https://www.kaggle.com/roshansharma/online-retail)\n
                    ---
                    For more information on Machine Learning, Statistics and Retail Analytics, check out these great resources:
                    - [Statquest for Statistics and ML](https://www.youtube.com/user/joshstarmer)
                    - [3Blue1Brown for Math](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
                    - [Sentdex for ML Coding Tutorials](https://www.youtube.com/user/sentdex)
                    - [Google Cloud Next'19 Keynote on Retail and AI](https://www.youtube.com/watch?v=pKEmQ1VMxsM)
                    - [DataRobot APAC Data Science's Presentation on Data Science in Retail](https://www.youtube.com/watch?v=PThNpfd3waE)\n
                    ---
                    This website was built using Python and the following libraries:
                    - [Streamlit] (https://docs.streamlit.io/en/stable/index.html)
                    - [ELI5](https://eli5.readthedocs.io/en/latest/overview.html)
                    - [Scikit-Learn](https://scikit-learn.org/)
                    - [Matplotlib](https://matplotlib.org/)
                    - [Seaborn](https://seaborn.pydata.org/)
                    - [Pandas](https://pandas.pydata.org/)
                    ''')
