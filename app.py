import streamlit as st
import pandas as pd
import pickle

courseCategory = ['Arts','Business','Programming','Health','Science']

pipe = pickle.load(open('pipe.pkl','rb'))

st.title('Course Completion Prediction')

category  = st.selectbox('Select the course category',sorted(courseCategory))

col1,col2 = st.columns(2)
with col1:
    timespent = st.number_input('Time Spent On Course in mins')

with col2:
    video_n = st.number_input('Number of video watched')

col3,col4 = st.columns(2)
with col3:
    quiz_n = st.number_input('NumberOfQuizzesTaken')

with col4:
    score = st.number_input('QuizScores')

completion  = st.number_input('Course Complete %')

if st.button('Predict Probability'):

    input_df = pd.DataFrame({'CourseCategory':[category],'TimeSpentOnCourse':[timespent],'NumberOfVideosWatched':[video_n],
                             'NumberOfQuizzesTaken':[quiz_n],'QuizScores':[score],'CompletionRate':[completion]})
    
    result = pipe.predict_proba(input_df)
    lossprob = result[0][0]
    winprob = result[0][1]

    # st.header('Course Completed '+"- "+str(round(winprob*100))+"%")
    st.header(f'According to our analysis, it is {str(round(winprob*100))}% likely that you have completed the course.')

