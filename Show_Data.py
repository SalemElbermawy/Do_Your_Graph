import streamlit as st
import pandas as pd
import plotly.express as ps
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

st.title("📊 Interactive Data Visualization App")
st.write("""
Upload your dataset and explore different types of plots for numerical and categorical data.
We will also handle **missing values** by imputing them automatically.
""")

data = st.file_uploader("📂 Upload Your CSV Data Here:")

if data is not None:
    st.write("✅ File uploaded successfully!")
    
    data = pd.read_csv(data)
    
    st.subheader("🔍 Preview of Uploaded Data")
    st.write(data.head())

    numerical_columns = data.select_dtypes(include="number").columns
    categorical_columns = data.select_dtypes(include="object").columns

    numerical_data = data[numerical_columns]
    categorical_data = data[categorical_columns]

    st.markdown("### 🔧 Handling Missing Values...")
    
    numerical_transforms_model = ColumnTransformer(transformers=[
        ("numerical", SimpleImputer(strategy="mean"), numerical_columns)
    ])
    numerical_data = numerical_transforms_model.fit_transform(numerical_data)
    numerical_data = pd.DataFrame(numerical_data, columns=numerical_columns)

    categorical_transform_model = ColumnTransformer(transformers=[
        ("categorical", SimpleImputer(strategy="most_frequent"), categorical_columns)
    ])
    categorical_data = categorical_transform_model.fit_transform(categorical_data)
    categorical_data = pd.DataFrame(categorical_data, columns=categorical_columns)

    st.success("Missing values have been filled successfully!")

    # Create tabs for visualization
    tab1, tab2, tab3 = st.tabs([
        "📈 Numerical Data Comparison",
        "📊 Distribution of Numerical Data",
        "🔠 Categorical Data Visualization"
    ])

    with tab1:
        st.subheader("📈 Compare Two Numerical Columns")
        st.write("Select two numerical columns to compare using **Line Plot** or **Scatter Plot**.")
        
        x = st.selectbox("Choose X-Axis:", options=numerical_columns)
        y = st.selectbox("Choose Y-Axis:", options=numerical_columns)
        plots = st.multiselect("Select Graph Type:", options=["Line_Plot", "Scatter_Plot"], default=None)
            
        if x is not None and y is not None:
            for plot_type in plots:
                if plot_type == "Line_Plot":
                    st.write(f"### 📈 Line Plot: {x} vs {y}")
                    graph = ps.line(x=numerical_data[x], y=numerical_data[y], title=f"Line Plot of {x} vs {y}")
                    st.plotly_chart(graph)
                elif plot_type == "Scatter_Plot":
                    st.write(f"### 🔵 Scatter Plot: {x} vs {y}")
                    graph = ps.scatter(x=numerical_data[x], y=numerical_data[y], title=f"Scatter Plot of {x} vs {y}")
                    st.plotly_chart(graph)

    with tab2:
        st.subheader("📊 Distribution of Numerical Data")
        st.write("Select a numerical column and choose whether to see its **Histogram** or **Box Plot**.")
        
        x = st.selectbox("Choose Column:", options=numerical_columns)
        plots = st.multiselect("Select Plot Type:", options=["Histogram", "Box"])
        
        if x is not None:
            for plot_type in plots:
                if plot_type == "Histogram":
                    st.write(f"### 📊 Histogram of {x}")
                    graph = ps.histogram(numerical_data, x=x, title=f"Histogram of {x}")
                    st.plotly_chart(graph)
                elif plot_type == "Box":
                    st.write(f"### 📦 Box Plot of {x}")
                    graph = ps.box(numerical_data, x=x, title=f"Box Plot of {x}")
                    st.plotly_chart(graph)

    with tab3:
        st.subheader("🔠 Categorical Data Visualization")
        st.write("Select a categorical column and choose **Bar Plot** or **Box Plot**.")
        
        x = st.selectbox("Choose Categorical Column:", options=categorical_columns)
        plots = st.multiselect("Select Plot Type:", options=["Bar Plot", "Box Plot"])
        
        if x is not None:
            for plot_type in plots:
                if plot_type == "Box Plot":
                    st.write(f"### 📦 Box Plot for {x}")
                    graph = ps.box(categorical_data, x=x, title=f"Box Plot of {x}")
                    st.plotly_chart(graph)
                elif plot_type == "Bar Plot":
                    st.write(f"### 📊 Bar Plot for {x}")
                    graph = ps.bar(categorical_data, x=x, title=f"Bar Plot of {x}")
                    st.plotly_chart(graph)
    