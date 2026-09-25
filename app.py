import streamlit as st
import joblib
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model

model = load_model("ann_model.keras")
feature = joblib.load("Feature.pkl")
Label = joblib.load("LabelEncoding.pkl")
preprocessing = joblib.load("Preprocessing.pkl")

st.set_page_config(page_title='Weather Classification',layout='centered',initial_sidebar_state='expanded')
st.title(" 🌦️ Weather Classification ")

st.divider()

with st.popover("ℹ️ About Weather Classification Project",use_container_width=True) :

    st.header("📖 About Project")

    st.write(
        """
        **Weather Classifiaction** is a deep learning
        application designed to predict the type of weather
        based on different environmental information.

        The application takes weather information such as
        temperature, humidity, wind speed, precipitation,
        atmospheric pressure, UV index, visibility, Cloud Cover as input.

        An Artificial Neural Network (ANN) is used to
        classify the input conditions into the expected
        weather type.
        """
    )

    st.divider()
    st.header("🌦️ Weather Classes")

    st.write("☀️ Sunny")
    st.write("☁️ Cloudy")
    st.write("🌧️ Rainy")
    st.write("❄️ Snowy")

    st.divider()

    st.header(" 🤖 Model Information")

    st.write("""   
    An **Artificial Neural Network (ANN)** is a deep learning model inspired by the way the human brain processes information.

    ANN consists of interconnected layers of neurons that learn
    patterns from the input data. In this project, the ANN learns
    relationships between different weather conditions and the
    corresponding weather type.
    """)

    st.write("**Task :** Weather Type Classification")
    st.write("**Framework :** TensorFlow / Keras")
    st.write("**Preprocessing :** Encoding + Scaling")

    st.header("📊 Model Comparison")
    Model_Comparison = pd.DataFrame({

    "Type of Model": [
        "Machine Learning ",
        "Machine Learning ",
        "Machine Learning ",
        "Machine Learning ",
        "Machine Learning ",
        "Deep Learning "
     ],
    "Model": [
        "Decision Tree",
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN",
        "Artificial Neural Network"
     ],
    "Test Accuracy (%)": [
        95.79,
        94.49,
        96.87,
        96.25,
        95.51,
        96.30
     ]
     })

    st.dataframe(
        Model_Comparison,
        hide_index=True,
        use_container_width=True
    )

    st.divider()
    st.header("🏆 Final Model Selection")

    st.write(
    """
    The **Artificial Neural Network (ANN)** was selected as the final model.

    The ANN achieved competitive performance on the weather classification
    task with **97.37% training accuracy, 97.16% validation accuracy
    and 96.30% test accuracy**.

    The ANN was therefore selected as the final model used in this
    Streamlit application for weather classification.
    """
    )

    st.subheader("📈 Train / Validation / Test Performance")

    score_df = pd.DataFrame({
    "Dataset": [
        "Training",
        "Validation",
        "Test"
    ],
    "Accuracy (%)": [
        97.37,
        97.16,
        96.30
    ],
    "Loss": [
        0.0815,
        0.1015,
        0.1230
    ]
      })

    st.dataframe(
    score_df,
    hide_index=True,
    use_container_width=True
    )

    st.subheader("🔢 Confusion Matrix")
    confusion_df = pd.DataFrame(
    [
        [416, 23, 3, 9],
        [9, 406, 2, 7],
        [3, 0, 432, 3],
        [4, 0, 2, 440]
    ],
    index=["Cloudy", "Rainy", "Snowy", "Sunny"],
    columns=["Cloudy", "Rainy", "Snowy", "Sunny"]
    )

    st.dataframe(
    confusion_df,
    use_container_width=True
    )    

    st.subheader("📋 Classification Report")
    classification_df = pd.DataFrame({
    "Weather Type": [
        "Cloudy",
        "Rainy",
        "Snowy",
        "Sunny"
    ],
    "Precision": [
        0.96,
        0.95,
        0.98,
        0.96
    ],
    "Recall": [
        0.92,
        0.96,
        0.99,
        0.99
    ],
    "F1-Score": [
        0.94,
        0.95,
        0.99,
        0.97
    ],
    "Support": [
        451,
        424,
        438,
        446
    ]
    })

    st.dataframe( 
    classification_df,
    hide_index=True,
    use_container_width=True
    )    

    st.write(
    """
    **Overfitting Check :** The training, validation and test accuracies
    are relatively close (97.37%, 97.16%, 96.30%), indicating no
    significant overfitting and good generalization to unseen data.
    """
    )
    st.divider()

    st.subheader("⚙️ How Does This Prediction Work ?")

    st.write(
        """
        The Prediction follows these steps :

        1. User enters weather information.
        2. Categorical features are encoded using OneHot Encoding.
        3. Numerical features are scaled using the Standard Scaler.
        4. The processed data is passed to the ANN model.
        5. The ANN predicts the weather class.
        6. The predicted class is displayed.
        """
    )

st.divider()
st.write(
    "Predict the weather type using an Artificial Neural Network "
    "based on the given weather conditions."
)

st.write(
    "Enter the current weather information below and click "
    "**Predict Weather Type**."
)
st.divider()

numeric_col = ['Temperature (°C)','Humidity (%)','Wind Speed (km/h)','Precipitation (%)','Atmospheric Pressure (hpa)','UV Index','Visibility (km)']

select_box = { 'Cloud Cover':['clear','partly cloudy','Overcast','cloudy'] }

example_inputs = {

    "Sunny Weather": {
        "Temperature (°C)": 30.0,
        "Humidity (%)": 45.0,
        "Wind Speed (km/h)": 8.0,
        "Precipitation (%)": 10.0,
        "Atmospheric Pressure (hpa)": 1020.0,
        "UV Index": 8.0,
        "Visibility (km)": 12.0,
        "Cloud Cover": "clear"
    },

    "Rainy Weather": {
        "Temperature (°C)": 18.0,
        "Humidity (%)": 85.0,
        "Wind Speed (km/h)": 15.0,
        "Precipitation (%)": 80.0,
        "Atmospheric Pressure (hpa)": 1005.0,
        "UV Index": 2.0,
        "Visibility (km)": 4.0,
        "Cloud Cover": "Overcast"
    },

    "Cloudy Weather": {
        "Temperature (°C)": 22.0,
        "Humidity (%)": 70.0,
        "Wind Speed (km/h)": 10.0,
        "Precipitation (%)": 35.0,
        "Atmospheric Pressure (hpa)": 1012.0,
        "UV Index": 4.0,
        "Visibility (km)": 8.0,
        "Cloud Cover": "partly cloudy"
    },

    "Snowy Weather": {
        "Temperature (°C)": -5.0,
        "Humidity (%)": 85.0,
        "Wind Speed (km/h)": 12.0,
        "Precipitation (%)": 70.0,
        "Atmospheric Pressure (hpa)": 1000.0,
        "UV Index": 2.0,
        "Visibility (km)": 3.0,
        "Cloud Cover": "Overcast"
     }
}

st.subheader("🧪 Try/Demo Example")

st.info(
    """ 
    1. Select an Example to Automatically Fill The Weather Values , Then Click **Predict Weather**.
    Demo examples are provided to help you understand the input format. """)
st.success("""
    2. For a more relevant prediction, select **Custom Input** and Enter 
      Your own  weather conditions.....
    """
)
st.write("")
selected_example = st.selectbox(
    "Select an example",
    ["Custom Input", "Sunny Weather", "Rainy Weather", "Cloudy Weather", "Snowy Weather"]
)

st.divider()

user = {}

for col in feature:

    if col in numeric_col:

        value = None

        if selected_example != "Custom Input":
            value = example_inputs[selected_example][col]

        user[col] = st.number_input(
            col,
            value=value,
            placeholder=f"Enter {col}",
            format="%.2f"
        )

    else:

        options = select_box[col]

        index = 0

        if selected_example != "Custom Input":
            index = options.index(example_inputs[selected_example][col])

        user[col] = st.selectbox(
            f"Select {col}",
            options,
            index=index
        )

st.divider()
Input = pd.DataFrame([user])

st.subheader(" 🗒️ Entered Weather Information ")

df = pd.DataFrame({"Weather Feature":Input.columns,"Entered Value":Input.values[0]})

st.table(df)

st.write("")
if st.button(" 🔮 Predict Weather ", use_container_width=True) :

    if Input.isnull().any().any() :

        st.warning("⚠️ Please Enter All Information")

    else :

        Input_preprocessed = preprocessing.transform(Input)

        Pred = model.predict(Input_preprocessed)

        st.divider()
        st.subheader("🎯 Prediction Result")

        weather_icons = {

                "Sunny": "☀️",

                "Cloudy": "☁️",

                "Rainy": "🌧️",

                "Snowy": "❄️"
            }
        weather = Label.inverse_transform([Pred.argmax()])[0]
        icon = weather_icons.get(weather, "🌦️")

        st.markdown(
        f"""
        <div style="
        text-align:center;
        ">
        <div style="font-size:60px;">{icon}</div>
        <h1>{weather}</h1>
        </div>
        """,
        unsafe_allow_html=True
        )

        confidence = (np.max(Pred[0]) * 100 )

        st.subheader("📊 Prediction Confidence")

        st.progress(int(confidence))

        st.write( f"**Confidence: {confidence:.2f}%**")

        st.subheader("📈 Class Probabilities")

        class_names = Label.classes_

        probability_df = pd.DataFrame({

                "Weather Type": class_names,

                "Probability (%)":Pred[0] * 100
            })

        probability_df["Probability (%)"] = (probability_df["Probability (%)"].round(2))

        st.bar_chart(probability_df.set_index("Weather Type"),use_container_width=True,x_label="Weather Type",y_label="Probability (%)",horizontal=True,height=250)

        st.dataframe(
                probability_df,
                hide_index=True,
                use_container_width=True
            )

        st.warning(
            "⚠️ **Important Note :** This prediction is generated by a deep learning model "
                    "and may not always be correct. The confidence score is model-estimated and "
                    "does not guarantee that the prediction is accurate.")
        st.info(
            "💡 **Tip :** If the prediction doesn't match the expected weather type, "
                    "try different input values and make 2–3 predictions to compare the results.")
        st.warning(
            "⚠️ **Note : This project is developed for educational and demonstration purposes. It is not a real-time weather forecasting system.**"
        )

st.divider()
st.caption(""" 🌦️ Weather Classification | ⚛ Deep Learning Project  
           🧠 Artificial Neural Network | 🧑‍💻 Author - Prerak Jasani """)
