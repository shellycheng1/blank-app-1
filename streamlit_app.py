import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("Glassdoor Gender Pay Gap.csv")

app_mode = st.sidebar.selectbox('Select a page >>', ["01 Introduction", "02 Data Overview", "03 Data Visualization", "04 Predictions", "05 Conclusions"])

if app_mode == "01 Introduction":
    st.title(":blue[Gender Pay Gap Analysis]")
    st.subheader("Providing insights into the gender pay gap based on salary data from Glassdoor")
    st. write ("The application can explore salary disparities across different job titles, education levels, and seniority levels.")

    st.image("20220307-gross-average-earningss-m-w.png")

    st.markdown('##### WHY THIS TOPIC?')
    st.markdown('We are exploring the topic of the gender pay gap as its understanding is essential for promoting equality, improving economic outcomes, and ensuring fairness in the workplace. By analyzing pay disparities across industries and demographics, we can identify systemic biases, encourage equal opportunities, and implement policies that promote fair compensation for all. Addressing the gender pay gap is crucial for creating a more equitable society, where everyone is compensated fairly for their contributions. Additionally, closing the gap can lead to higher productivity, greater job satisfaction, and a stronger, more inclusive economy, benefiting everyone.')
    st.markdown("##### OUR GOAL")
    st.markdown("Our goal is to analyze the gender pay gap by examining factors such as Job Title, Gender, Age, Education, Department, Seniority, Base Pay, and Salary Bonus. By exploring these variables, we aim to uncover the key drivers behind wage disparities and identify potential biases. This will help inform policies, promote fair compensation practices, and support efforts to close the gender pay gap, ultimately fostering equality and improving workplace dynamics.")
    st.markdown("##### OUR DATA")
    st.markdown("Our dataset is derived from data collected by Glassdoor, a employment listing and search site. It involves several variables that factor into the pay gap:")
    st.markdown(":red[Job Title]: Occupation and position of the individual (Graphic Designer, Software Engineer, Warehouse Associate, etc.)")
    st.markdown(":red[Gender]: Classified between Male and Female in this dataset.")
    st.markdown(":red[Age]: Age in years, from 18 to 65.")
    st.markdown(":red[PerfEval]: Performance Evaluation Score, from 1 to 5.")
    st.markdown(":red[Education]: Level of Education (High School, College, Masters, PhD)")
    st.markdown(":red[Dept]: Department of the position (Administration, Operations, Sales, Engineering, Management)")
    st.markdown(":red[Seniority]: Number of years worked")
    st.markdown(":red[Base Pay]: Annual basic pay in dollars ($).")
    st.markdown(":red[Bonus]: Annual bonus pay in dollars ($).")

    st.markdown('##### MISSING VALUES/ COMPLETENESS')
    st. write ("Our dataset has no missing or null values, allowing it to have a 100 percent completeness ratio.")

    
## Data Overview
elif app_mode == "02 Data Overview":

    st.title("Data Overview")

    st.subheader("Dataset Overview")
    st.dataframe(df.head())


    # Basic statistics
    st.subheader("Summary Statistics")
    st.write(df.describe())


    # Gender-based salary comparison
    st.subheader("Salary Distribution by Gender")
    fig, ax = plt.subplots()
    df.groupby("Gender")["BasePay"].mean().plot(kind="bar", ax=ax, color=["pink", "blue"])
    ax.set_ylabel("Average Base Pay")
    st.pyplot(fig)


    # Seniority level vs Salary
    st.subheader("Seniority Level vs Salary")
    fig, ax = plt.subplots()
    df.groupby("Seniority")["BasePay"].mean().plot(kind="line", marker="o", ax=ax)
    ax.set_xlabel("Seniority Level")
    ax.set_ylabel("Average Base Pay")
    st.pyplot(fig)


    # Job title salary comparison
    st.subheader("Top 5 Job Titles with Highest Pay")
    top_jobs = df.groupby("JobTitle")["BasePay"].mean().nlargest(5)
    st.bar_chart(top_jobs)


## Data Visualization

# Main Data Visualization logic
elif app_mode == "03 Data Visualization":
   
    # Sidebar for job title selection
    select = st.sidebar.selectbox('Select Job Title', df['JobTitle'].unique())
    gender_data = df[df['JobTitle'] == select]  # Filter the data based on the selected job title


    # Display the selected Job Title in the markdown header for all charts
    st.title(f"**Data Visualization for {select}**")


    # Bar Chart
    list_of_var = df.columns
    st.markdown("##### Bar Chart")
    user_selection = st.selectbox("Select a variable", list_of_var)


    st.bar_chart(gender_data[user_selection])  # Apply job title filter to this chart


    # Bar Plot (BasePay vs Gender)
    st.markdown("##### Bar Plot")
    user_selections = st.multiselect("Select a variable", list_of_var, ["BasePay", "Gender"])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(y=user_selections[0], x=user_selections[1], data=gender_data, palette='coolwarm_r')  # Apply job title filter
    st.pyplot(fig)


    # Scatterplot (BasePay vs Age)
    st.markdown("##### Scatterplot")
    user_selections1 = st.multiselect("Select a variable", list_of_var, ["BasePay", "Age"])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(y=user_selections1[0], x=user_selections1[1], data=gender_data, hue='Gender', palette='coolwarm_r')  # Apply job title filter
    st.pyplot(fig)


    st.sidebar.checkbox("Show Analysis by Education Level", True, key="education_checkbox")
    select_status = st.sidebar.radio("Education", ('High School', 'College', 'Masters', 'PhD'))


    # Function to count the cases by Education category
    def get_total_dataframe(dataset, education_column='Education'):
        # Ensure the Education column is of categorical type with a defined order
        ordered_categories = ['High School', 'College', 'Masters', 'PhD']
        dataset[education_column] = pd.Categorical(dataset[education_column], categories=ordered_categories, ordered=True)


        st.write("Unique values in the 'Education' column:", dataset[education_column].unique())


        # Count the occurrences of each education level
        education_counts = dataset[education_column].value_counts()


        # Create a DataFrame from the dictionary
        education_counts_dict = {category: education_counts.get(category, 0) for category in ordered_categories}
       
        total_dataframe = pd.DataFrame({
            'Education': list(education_counts_dict.keys()),
            'Number of cases': list(education_counts_dict.values())
        })


        return total_dataframe


    if st.sidebar.checkbox("Show Education Level Analysis", True, key="analysis_checkbox"):
        gender_total = get_total_dataframe(gender_data)


        # Display the gender_total DataFrame for debugging
        st.write("Gender total DataFrame:", gender_total)


        # Check if gender_total is empty or doesn't have the required columns
        if not gender_total.empty and 'Education' in gender_total.columns and 'Number of cases' in gender_total.columns:
            st.markdown(f"### **Base Pay by Education for {select}**")  # Display Job Title in the title
       


            if not st.checkbox('Hide Graph', False, key="hide_graph_checkbox"):
                # Group bars by gender
                state_total_graph = px.bar(gender_data,
                                           x='Education',
                                           y='BasePay',
                                           color='Gender',  # Divides the bars by Gender
                                           barmode='group',  # Groups the bars side-by-side
                                           labels={'BasePay': f'Base Pay in {select}'},
                                           color_discrete_sequence=[ "#e8334b", "#037ffc"],  # Custom color palette
                                           category_orders={'Education': ['High School', 'College', 'Masters', 'PhD']})  # Order the Education categories
                st.plotly_chart(state_total_graph)
        else:
            st.error("The gender_total DataFrame is empty or does not have the required columns.")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Predictions
if app_mode == "04 Predictions":

    # Display basic dataset info
    # Enlarging "Predictions" title using HTML & CSS
    st.markdown("<h1 style='font-size:40px;'>Predictions</h1>", unsafe_allow_html=True)


    # Making "Dataset Preview" slightly smaller
    st.markdown("<h3 style='font-size:25px;'>Dataset Preview</h3>", unsafe_allow_html=True)


    # Displaying dataset in a scrollable table
    st.dataframe(df.head(20))


    # Drop irrelevant columns (adjust based on dataset structure)
    df = df[['Gender', 'BasePay', 'JobTitle', 'Dept', 'Seniority', 'Age', 'PerfEval', 'Education', 'Bonus']]  # Adjust as needed


    # Handle missing values
    df.dropna(inplace=True)


    # Encode categorical variables
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # Assuming 0 = Female, 1 = Male
    df['JobTitle'] = le.fit_transform(df['JobTitle'])
    df['Dept'] = le.fit_transform(df['Dept'])


    # Extract original unique education levels before encoding
    education_labels = df['Education'].unique().tolist()
    df['Education'] = le.fit_transform(df['Education'])  # Encode after extracting labels


    # Main page content
    st.markdown("<h3 style='font-size:22px;'>Select Input Variables</h3>", unsafe_allow_html=True)


    # Gender selection
    gender_map = {0: "Female", 1: "Male"}
    gender = st.selectbox(f"Select Gender", list(gender_map.values()))


    # Age range selection
    age_ranges = ["18-25", "26-35", "36+"]
    age = st.selectbox("Select Age Range", age_ranges)


    # Education level selection
    education = st.selectbox("Select Education Level", education_labels)


    # Bonus range selection
    bonus_ranges = ["0-5K", "5K-10K", "10K-20K", "20K+"]
    bonus = st.selectbox("Select Bonus Range", bonus_ranges)


    # Model settings
    st.markdown("<h3 style='font-size:22px;'>Model Settings</h3>", unsafe_allow_html=True)
    train_size = st.number_input("Train Set Size", min_value=0.1, max_value=0.9, step=0.05, value=0.7)


    # Feature selection
    st.markdown("<h3 style='font-size:22px;'>Select Explanatory Variables</h3>", unsafe_allow_html=True)
    all_features = df.drop(columns=['BasePay'])  # Exclude target variable from selection
    selected_features = st.multiselect("Select Features", all_features.columns, default=all_features.columns.tolist())


    # Ensure at least one feature is selected
    if not selected_features:
        st.warning("Please select at least one explanatory variable.")
    else:
        # Define X (independent variables) and y (Base Pay as target variable)
        X = df[selected_features]
        y = df['BasePay']


        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=(1 - train_size), random_state=42)


        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)


        # Make predictions
        y_pred = model.predict(X_test)


        # Model evaluation
        st.subheader("🎯 Model Results")
        st.write(f"1) **Explained Variance:** {explained_variance_score(y_test, y_pred) * 100:.2f}%")
        st.write(f"2) **Mean Absolute Error (MAE):** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"3) **Mean Squared Error (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"4) **R-Squared Score:** {r2_score(y_test, y_pred):.2f}")


        # Visualization options for the user to select
        st.markdown("<h3 style='font-size:22px;'>Select Visualization Type</h3>", unsafe_allow_html=True)
        visualization_choice = st.selectbox(
            "Choose a Visualization",
            ["Base Pay Distribution by Gender", "Predictions vs Actual Values", "Residuals Plot"]
        )


    if visualization_choice == "Base Pay Distribution by Gender":
        st.subheader("📊 Base Pay Distribution by Gender")
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df['Gender'], y=df['BasePay'], palette='coolwarm')
        plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
        plt.xlabel('Gender')
        plt.ylabel('Base Pay')
        plt.title('Base Pay Distribution by Gender')
        st.pyplot(plt)
    elif visualization_choice == "Predictions vs Actual Values":
        st.subheader("📈 Predictions vs Actual Values")
        plt.figure(figsize=(8, 5))
        plt.plot(y_test.values, label="Actual", color='blue')
        plt.plot(y_pred, label="Predicted", color='red')
        plt.title("Actual vs Predicted Base Pay")
        plt.xlabel("Sample Index")
        plt.ylabel("Base Pay")
        plt.legend()
        st.pyplot(plt)
    elif visualization_choice == "Residuals Plot":
        st.subheader("📉 Residuals Plot")
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, residuals, color='green')
        plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red', linestyles='dashed')
        plt.title("Residuals Plot")
        plt.xlabel("Actual Base Pay")
        plt.ylabel("Residuals (Actual - Predicted)")
        st.pyplot(plt)

#Conclusion page

if app_mode == "05 Conclusions":
    st.title(":blue[Gender Pay Gap Conclusion]")
    
    st.write(
        """
        Our analysis provides important insights into the gender pay gap across various job categories. 
        Overall, the data indicates that women tend to earn less than men in most job sectors, both in the raw data 
        and in the predictive modeling results.
        """
    )
    
    st.subheader("Data Quality")
    st.write(
        """
        The dataset we used is large and robust, comprising a diverse range of job categories and demographic data. 
        The data includes both numerical and categorical variables, allowing for a comprehensive understanding of 
        pay disparities. However, we recognize that some factors—such as biases in reporting, limited data for specific industries, 
        or unequal representation of genders in certain roles—could impact the accuracy of the conclusions drawn. 
        Still, overall, the data quality supports the analysis and gives us a strong foundation for insights.
        """
    )
    
    st.subheader("Model Related Improvements")
    st.write(
        """
        While logistic regression has been an effective model for identifying and predicting trends in gender-based pay disparities, 
        future improvements could include experimenting with more complex models such as random forests or gradient boosting. 
        These models may help in capturing more intricate relationships between variables. Additionally, 
        feature engineering and tuning could be optimized to enhance model accuracy.
        """
    )
    
    st.subheader("Key Insights")
    st.write(
        """
        - **Average Pay Gap**: On average, men earn more than women across most job categories, with the gap being more pronounced 
        in certain sectors. This finding aligns with the widely observed global trend of gender-based wage disparities.
        
        - **Disparities by Role**: The gender pay gap varies by job category, with some industries showing more significant disparities 
        than others. However, even in more gender-balanced fields, women tend to earn less on average than their male counterparts.
        
        - **Predictive Trends**: Our predictions, based on the current data, show that women’s pay continues to lag behind men’s 
        even after accounting for factors such as experience, education, and job title. This suggests that gender bias might still be at play.
        
        - **Potential Biases**: While our model adjusts for various factors, there is still potential for overlooked biases. 
        Future data collection and model refinements could further explore the impact of implicit biases, regional differences, and 
        other factors that might contribute to the gender pay gap.
        """
    )
    
    st.write(
        """
        In conclusion, the data highlights and predicts a significant 
        gender pay gap that warrants attention. Further studies and continued efforts toward equality, as well as ongoing 
        improvements in data collection and predictive modeling, are necessary steps toward closing this gap and ensuring fairer 
        compensation for all employees, regardless of gender.
        """
    )

    st.subheader("Group Members")
    st.write("Shelly Cheng, Kendra Contreras, Andrea Pelaez, Falisha Khan")
