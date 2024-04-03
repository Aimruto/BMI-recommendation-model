import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


data = pd.read_csv('C:/Users/KIIT/Documents/new-minor/Data/input.csv')

lbl_enc = LabelEncoder()
data.iloc[:, 0] = lbl_enc.fit_transform(data.iloc[:, 0])


std_sc = StandardScaler()
df = pd.DataFrame(data)
df.iloc[:, 1:-1] = std_sc.fit_transform(df.iloc[:, 1:-1])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
rfc.fit(X, y)

recommendations_df = pd.read_csv('C:/Users/KIIT/Documents/new-minor/Data/recom.csv')

def health_test(gender, height, weight):
    '''Input gender as Male/Female, height in cm, weight in Kg'''
    individual_data_dict = {'Gender': [gender], 'Height': [height], 'Weight': [weight]}
    individual_data = pd.DataFrame(data=individual_data_dict)

    individual_data.iloc[:, 0] = lbl_enc.transform(individual_data.iloc[:, 0])

    individual_data.iloc[:, 1:] = std_sc.transform(individual_data.iloc[:, 1:])

    y_pred = rfc.predict(individual_data)

    label_mapping = {
        0: 'Extremely Weak',
        1: 'Weak',
        2: 'Normal',
        3: 'Overweight',
        4: 'Obesity',
        5: 'Extreme Obesity'
    }
    health_status = label_mapping[y_pred[0]]

    recommendations = get_recommendations(health_status, gender)

    return health_status, recommendations

def get_recommendations(health_status, gender):
    recommendations = recommendations_df[(recommendations_df['Health Category'] == health_status) & 
                                          (recommendations_df['Gender'] == gender)]
    exercise_recommendation = recommendations['Exercise Recommendation'].iloc[0]
    diet_recommendation = recommendations['Diet Recommendation'].iloc[0]
    return exercise_recommendation, diet_recommendation

st.set_page_config(page_title="BMI Health Recommendation")

# Streamlit UI
st.title("BMI Health Recommendation System")
st.markdown(
    """
    <style>
    
        .stButton>button {
            background-color: #7353DF;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            border: none;
        }
    </style>
    """, unsafe_allow_html=True)

gender = st.radio("Select Gender", ("Male", "Female"))
height = st.number_input("Enter Height (cm)", min_value=0)
weight = st.number_input("Enter Weight (kg)", min_value=0)

if st.button("Get Recommendations"):
    if not gender or not height or not weight:
        st.warning("Please fill in all the input fields.")
    else:
        health_status, recommendations = health_test(gender, height, weight)
        st.write(f"### Health Status: {health_status}")
        st.write(f"### Exercise Recommendation: {recommendations[0]}")
        st.write(f"### Diet Recommendation: {recommendations[1]}")

        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)
        st.write(f"### BMI: {bmi:.2f}")
        st.write(f"### Gender: {gender}")
