import streamlit as st
import numpy as np
from urllib import request
import json, boto3
from datetime import datetime

# Get the current date and time
now = datetime.now()

# Format the date and time as a string
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")

# Server address 
fargate = "35.170.76.218"
url = f"http://{fargate}:9910/audioApp/"
# url = "http://localhost:9910/audioApp/"

s3_client = boto3.client('s3',aws_access_key_id = st.secrets["aws"]["aws_access_key_id"],
                    aws_secret_access_key = st.secrets["aws"]["aws_secret_access_key"])

st.set_page_config(page_title="Audio 48",page_icon="⚽",initial_sidebar_state="expanded")


def call_service(endpoint,data):
    # Prep json inputs to call MATLAB code 
    headers = {"Content-Type": "application/json"}
    body = json.dumps({"nargout": 1, "rhs" : [data]})
    # Convert to string and encode
    body = str(body)
    body= body.encode("utf-8")
    # Post method
    req = request.Request(url+endpoint, data=body, headers=headers)
    # Response
    resp = request.urlopen(req)
    result = json.loads(resp.read())
    return result

@st.cache(allow_output_mutation=True)
def upload_file(file):
    bytes_data = file.getvalue()
    filename = f'data_{date_time}.wav'
    with open(filename,'wb') as f:
        f.write(bytes_data)
    # st.write(bytes_data)
    # s3_client.upload_file(file_name, bucket, object_name)
    s3_client.upload_file(filename,"audio48","unknown/"+filename)
    data = "https://audio48.s3.amazonaws.com/unknown/"+filename
    return data

st.title('Messi or Mbappe ⚽')

# st.write("Date and time:", date_time)

mode = st.radio("Input method",('upload','list','default'))

if mode == 'upload':
    file = st.file_uploader("Upload file")
    if file:
        st.audio(file)
        data = upload_file(file)
elif mode == 'list':
    f = [key['Key'] for key in s3_client.list_objects(Bucket='audio48',Prefix="unknown/")['Contents']]
    d = st.selectbox("Select Data",f,
            format_func = lambda x : x.replace("unknown/","").replace(".wav",""))
    # st.write(d)
    data = "https://audio48.s3.amazonaws.com/"+d
else:
    data = "https://audio48.s3.amazonaws.com/unknown/data.wav"
    if st.button('Download'):
        request.urlretrieve(data,filename='data.wav')


pre = st.checkbox("Preprocess")
if pre:
    result = call_service(endpoint="audioPipeline",data=data)
    # st.write(result)
    data = result['lhs'][0]['mwdata'][0]['mwdata']
    size = result['lhs'][0]['mwdata'][0]['mwsize']
    X = np.transpose(np.array(data).reshape(64,96))
    st.write(X)
    st.write('shape',X.shape)
    inf = st.button("Predict")
    if inf:     
        result = call_service(endpoint="predFcn",data=X.tolist())
        # st.write(result)
        Y = result['lhs'][0]['mwdata']
        # st.write(Y)
        labels = ["mbappe","messi"]
        id = np.argmax(Y)
        st.sidebar.subheader(labels[id])
        if id == 0:
            st.sidebar.image('img/mbappe.jpg')
        else:
            st.sidebar.image('img/messi.jpg')