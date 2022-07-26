



import streamlit as st
import pandas as pd
import sweetviz
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
#----------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title="The Sweetviz App", layout="wide")

#---------------------------------#
# Model building
def build_model(df):
    df = df.loc[:] # Having a look at the all column
    X = df.iloc[:, :-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y
    
    st.markdown('**1.2. Dataset dimension**')
    st.write('X')
    st.info(X.shape)
    st.write('Y')
    st.info(Y.shape)

    st.markdown('**1.3. Variable details**:')
    st.write('X variable (All are shown)')
    st.info(list(X.columns[:, :-1]))
    st.write('Y variable')
    st.info(Y.name)
    
    # Using sweetviz model
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
    predictions_train = (X_train, X_train, Y_train, Y_train)
    predictions_test = (X_train, X_test, Y_train, Y_test)

    # Analyzing the dataset using Sweetviz library
    train_report = sweetviz.analyze([predictions_train, "Train"], target_feat=Y)
    comparison_report = sweetviz.compare([predictions_train, "Train"], [predictions_test, "Test"], target_feat=Y)
    
    st.subheader("2. The Sweetviz report")
    
    train_html = train_report.show_html("Report.html"), unsafe_allow_html=True
    compare_html = comparison_report.show_html("Comparison.html"), unsafe_allow_html=True
    
    # Creating a whole report in the form of HTML file
    st.write("Train set Sweetviz report")
    st.markdown(filedownload(train_html)
    
    # Comparing the training and testing dataset using Sweetviz
    st.write("Comparing Train & Test dataset Sweetviz report")
    st.markdown(filedownload(comparison_html)
    
# Download html file data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href
    
    
#---------------------------------#
st.write("""
# The Sweetviz App

**Sweetviz** library is used to generate beautiful, high-density, highly detailed visualization to Exploratory Data Analysis.

Developed by: [Akshay Narvate](https://akshaynarvate-resume-app-streamlit-app-y86511.streamlitapp.com/)
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""
[Example CSV input file](https://github.com/akshaynarvate/Future-Sales-Prediction/blob/main/future_sales_predicton.csv)
""")

#---------------------------------#
# Main panel

# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        #X = pd.DataFrame(boston.data, columns=boston.feature_names)
        #Y = pd.Series(boston.target, name='response')
        X = pd.DataFrame(boston.data, columns=boston.feature_names).loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        Y = pd.Series(boston.target, name='response').loc[:100] # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('The Boston housing dataset is used as the example.')
        st.write(df.head(5))

        build_model(df)
