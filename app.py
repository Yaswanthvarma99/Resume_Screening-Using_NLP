import streamlit as st
import pickle
import re
import nltk
from summarizer import Summarizer

nltk.download('punkt')
nltk.download('stopwords')
image = "/Users/yashuvaishu/Desktop/resumeimg.jpeg"
#loading models
model = pickle.load(open('model.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text


# web app
def main():
    
    st.title("Resume Screening App")
    st.image(image=image)
    uploaded_file = st.file_uploader('Upload Resume', type=['txt'])

    if uploaded_file is not None:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('latin-1')
        
           

            word_vectorizer = pickle.load(open('word_vectorizer.pkl','rb'))
            clean = clean_resume(resume_text)
            input_features = word_vectorizer.transform([clean])
            prediction_id =  model.predict(input_features)[0]
            st.write(prediction_id)
            
           

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id)

            st.write("Predicted Category:", category_name)
            bert_summarizer = Summarizer()
            summary = bert_summarizer(resume_text)
            st.write(summary)



# python main
if __name__ == "__main__":
    main()
    
