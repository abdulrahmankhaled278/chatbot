import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from pinecone import PodSpec, Pinecone, ServerlessSpec
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import speech_recognition as sr
import whisper
import re
from google.cloud import texttospeech
import time  

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

if st.sidebar.button("Load whisper model"):
    st.session_state.model = load_whisper_model()
    st.success("Model loaded successfully")

def load_document(file):
    print("Loading PDF document: " + file)
    loader = PyPDFLoader(file)
    data = loader.load()
    return data

def chunk_data(data, chunk_size, chunk_overlap=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    chunks = text_splitter.split_documents(data)
    return chunks


def delete_pinecone_index(index_name='all') :
    pc = Pinecone()
    if index_name == 'all' :
        indexes = pc.list_indexes().names()
        print(f"Deleting all indexes ...")
        for index in indexes :
            pc.delete_index(index)
        print("done")
    else :
        print(f"Deleting index {index_name} ...")
        pc.delete_index(index_name)
        print("done")

def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec
    pc = pinecone.Pinecone()
    
    
    
    # Use the 'text-embedding-3-large' model with 3072 dimensions
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    # Check if the index already exists
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # Create a new index with the correct dimension
        print(f'Creating index {index_name} and embeddings ...', end='')
        
        # Creating a new index with the correct dimensions for 'text-embedding-3-large'
        pc.create_index(
            name=index_name,
            dimension=3072,  # Set to the dimension size of the embedding model
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # Process the input documents, generate embeddings, and insert them into the index
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store
## *********************************** few shot ************************************************************** ##
examples = [
    {
        "question": "مين هي؟ / انتم مين؟",
        "answer": """
        معاك يا فندم شركة ممفيس للسياحة أحد الشركات الرائدة للسياحة والسفر - تأسست عام 1955 - (ترخيص سياحة أ ) فى جمهورية مصر العربية, والمتخصصة في تقديم أمثل خدمات السفر والعروض السياحية بأسعار مميزة للأفراد والشركات والمؤسسات.
        
        حصلت شركة ممفيس على جائزة التميز للعام السادس على التوالى والمقدمة من موقع تريب أدفايزر العالمى لتقييم خدمات شركات السياحة المقدمة للعملاء فى جميع أنحاء العالم، أكثر من 9440 تقرير عن خدماتنا.
        
        بإمكاننا دائما تقديم المساعدة فى التخطيط لرحلاتك كون عروضنا السياحية هى الأكثر إبتكارا والأفضل أسعاراً ، حيث تتنوع خدماتنا لتشمل حجوزات الطيران الفنادق، النقل وإصدار التأشيرات.
        """
    },
    {
        "question": "صفحاتكم على السوشيال ميديا؟ / اكاونت السوشيال ميديا؟",
        "answer": """
        حضرتك تقدر تتواصل معانا عن طريق صفحتنا على فيسبوك من خلال الرابط ده
        https://www.facebook.com/MemphisReligiousTourism
        """
    },
    {
        "question": "أتصل بيكم إزاي؟ / رقم تواصل؟ / اتواصل معاكم ازاي؟",
        "answer": """
        للتواصل: حضرتك تقدر كلمنا على اي رقم من دول 01014445674 - 01012203070 او ابعتلنا رسالة على الواتس اب https://wa.me/message/D2MPSJS2GGKGG1
        """
    },
    {
        "question": "أرقام الفروع؟",
        "answer": """
        المقر الرئيسي: ٢٤ ش مراد، الجيزة
        https://maps.app.goo.gl/5W5qKETiEzkoNSVD8?g_st=iw
        
        فرع العبور: ٣٤ عمارات العبور، صلاح سالم
        https://maps.app.goo.gl/YzNruDs37gM7h9rb6?g_st=iw
        
        فرع حلوان: ٤٢ ش مصطفى فهمي، حلوان
        https://maps.app.goo.gl/jj498kNVZmjzqJqR7?g_st=iw
        """
    },
    {
        "question": "مواعيد الفروع / مواعيدكم؟ / الفرع شغال امتى؟ / الفرع بيفتح امتي؟",
        "answer": """
        ايام العمل من يوم الاحد الى الخميس - ساعات العمل من 10 ص حتى 6 م
        """
    },
    {
        "question": "فروعكم فين؟ / أماكنكم فين؟",
        "answer": """
        المقر الرئيسي: ٢٤ ش مراد، الجيزة
        فرع العبور: ٣٤ عمارات العبور، صلاح سالم
        فرع حلوان: ٤٢ ش مصطفى فهمي، حلوان
        """
    },
    {
        "question": "اللوكيشن / الخريطة / عنوانكم على maps؟",
        "answer": """
        حضرتك تقدر تشرفنا في أي فرع من دول داخل القاهره
        المقر الرئيسي: ٢٤ ش مراد، الجيزة
        https://maps.app.goo.gl/5W5qKETiEzkoNSVD8?g_st=iw
        
        فرع العبور: ٣٤ عمارات العبور، صلاح سالم
        https://maps.app.goo.gl/YzNruDs37gM7h9rb6?g_st=iw
        
        فرع حلوان: ٤٢ ش مصطفى فهمي، حلوان
        https://maps.app.goo.gl/jj498kNVZmjzqJqR7?g_st=iw
        """
    },
    {
        "question": "ممكن لوكيشن فندق مكة؟",
        "answer": """
        (حسب البرنامج)
        """
    },
    {
        "question": "ممكن لوكيشن فندق المدينة؟",
        "answer": """
        (حسب البرنامج)
        """
    },
    {
        "question": "هل ليكم فروع في المحافظات؟",
        "answer": """
        لينا فروع في المحلة والغردقة
        """
    },
    {
        "question": "سعر / بكام / تكلفة المنتج؟",
        "answer": """
        يعتمد سعر او تكلفة المنتج حسب البرنامج المعلن وحسب تفضيلات العميل
        """
    },
    {
        "question": "الأسعار للحج / الاسعار لرحلة العمرة / سعر الرحلة كام الرحلة بكام؟",
        "answer": """
        الأسعار:
        - 79 الف للفرد في الغرفة الرباعية
        - 95 الف للفرد في الغرفة الثلاثية
        - 125 الف للفرد في الغرفة الثنائية
        """
    },
    {
        "question": "الرحلة للعمرة العمرة امتى؟ / مدة العمرة؟",
        "answer": """
        السفر 16 رمضان الموافق 26 مارس - العودة 30 رمضان الموافق 9 ابريل (قبل العيد)
        هنقضي 4 ليالي في المدينة وبعد كدا نختم العشر الاواخر في مكة
        """
    },
    {
        "question": "الطيران خطوط ايه؟",
        "answer": """
        الطيران يا فندم بيكون عن طريق خطوط مصر للطيران
        """
    },
    {
        "question": "إيه الفنادق في رحلة العمرة؟",
        "answer": """
        فندق المدينة: جراند بلازا - 3 نجوم - صف ثاني - 3-5 دقائق مشي من المسجد النبوي
        فندق مكة: فوكو VOCO - 5 نجوم بالمواصلات - الباص بينزل حضرتك على مسافة 250 م من الحرم
        """
    },
    {
        "question": "فندق المدينة فين؟",
        "answer": """
        فندق المدينة في المركزية الشمالية - يبعد 3-5 دقائق مشي من المسجد النبوي
        """
    },
    {
        "question": "فندق مكة فين؟",
        "answer": """
        فندق مكة في المسفلة على شارع إبراهيم الخليل 1,3 كم من الحرم - الفندق بيوفر باصات لنقل المعتمرين من وإلى الحرم مسافة 250 م من الحرم - خدمة على مدار الساعة
        """
    },
    {
        "question": "عايز فندق قريب من الحرم / سعر فندق قريب من الحرم كام؟",
        "answer": """
        متوفر برنامج على بلاط الحرم
        تفاصيل البرنامج نفس الرحلة المعلنة باختلاف الفنادق فقط
        فندق المدينة: الايمان الحرم
        فندق مكة: درر الايمان - أبراج الصفوة
        سعر البرنامج للفرد في الغرفة الثنائية 325 الف جنيه
        """
    },
    {
        "question": "هل الرحلة ختام مكة؟",
        "answer": """
        ايوه يا فندم الرحلة ختام مكة
        """
    },
    {
        "question": "هتروحوا امتى؟ / السفر امتى؟",
        "answer": """
        السفر يوم 16 رمضان الموافق 26 مارس ان شاء الله
        """
    },
    {
        "question": "انتو سعركم غالي كده ليه؟",
        "answer": """
        نتيجة غلو الفنادق ومحاولتنا للحفاظ على الخدمة بسعر معقول نسبيًا لفترة العشر الاواخر. بنتمنى نقدر نخدمك في رحلة قريبة بعد رمضان
        """
    },
    {
        "question": "طب انا معايا تأشيرة / التأشيرة بتخصم كام؟",
        "answer": """
        التأشيرة التجارية، السياحية، العائلية، الحكومية بنخصم لها 9000 جنيه مقابل التأشيرة والباركود. التأشيرة الشخصية بنخصم لها 5000 مقابل التأشيرة.
        """
    }
]

from langchain.prompts import FewShotPromptTemplate

def create_few_shot_prompt(examples, input_variable):
    example_prompts = []
    for example in examples:
        example_prompts.append(f"Q: {example['question']}\nA: {example['answer'].strip()}")

    example_prompts_str = "\n\n".join(example_prompts)

    return f"""
            You are an enthusiastic and helpful customer service representative for an Egyptian tourism company called Memphis. 
            Your role is to provide information in the local Egyptian dialect, using casual, friendly, and conversational language. 
            Always be concise and directly address the user's specific questions, based on the examples provided. 
            Answer confidently as if you have full knowledge of Memphis's offerings. 
            Only provide the information relevant to the question, without adding any extra details unless explicitly asked. 
            Avoid using phrases like 'According to the text provided' or 'في السياق المقدم'. 
            Do not mention limitations in the text or suggest that you are retrieving answers from a document. Instead, use your knowledge to provide informative, confident responses. If necessary, make a helpful and educated guess based on typical Memphis offerings, or invite the user to provide more specifics so you can assist further.
            Your goal is to ensure that the user feels well-informed, supported, and satisfied with concise answers at all times.
            Answer in a way that is personal, confident, and informative. If specific details are missing, provide a brief and educated guess based on typical Memphis offerings, or politely ask for more specifics.
            Your goal is to ensure that the user feels well-informed, supported, and satisfied with concise answers at all times.
    Examples:
    {example_prompts_str}

    Now, answer the following question:

    Q: {{{input_variable}}}
    """

# Create the prompt template using examples
few_shot_prompt_template = create_few_shot_prompt(examples, "content")

def initialize_memory_and_chains(vector_store):
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        llm = ChatOpenAI(model='gpt-4o', temperature=0.6)

        prompt = ChatPromptTemplate(
            input_variables=["content", "chat_history"],
            messages=[
                SystemMessage(content=few_shot_prompt_template),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{content}")
            ]
        )

        retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

        st.session_state.crc_document = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            chain_type='stuff',
            memory=st.session_state.memory,
            verbose=False
        )

        st.session_state.chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=st.session_state.memory,
            verbose=False
        )


def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_answer(text) :
    return re.sub(r'#', '', text)

def ask_context_aware(vector_store, q, k=3):
    initialize_memory_and_chains(vector_store)
    
    lack_of_knowledge_phrases = ["I don't have","I don't know","لا أعلم", "ليس لدي", "آسف", "لا أملك", "لا اعرف","لا استطيع"]
    
    chain = st.session_state.chain
    crc_document = st.session_state.crc_document

    similarity_score = vector_store.similarity_search_with_score(q)[0][1]
    print(f"Similarity score: {similarity_score}")
    
    def invoke_llm_directly(query):
        llm = ChatOpenAI(model='gpt-4o', temperature=1)
        return llm.invoke(query).content

    # Record the start time
    start_time = time.time()
    
    if similarity_score < 0.3:
        st.write("No similar documents found, invoking LLM directly.")
        response = chain.invoke({'content': q})
        response_text = response['text']
        
        if any(phrase in response_text for phrase in lack_of_knowledge_phrases):
            st.write("Lack of Knowledge detected, invoking LLM directly.")
            # Record the end time just before starting the response
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return invoke_llm_directly(q)
        else:
            print(response)
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return response_text
    else:
        st.write("Similar documents found, invoking CRC.")
        response = crc_document.invoke({'question': q})
        response_answer = response['answer']
        
        if any(phrase in response_answer for phrase in lack_of_knowledge_phrases):
            st.write("Lack of Knowledge detected, invoking LLM directly.")
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return invoke_llm_directly(q)
        else:
            print("No Lack of Knowledge detected")
            print(response)
            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_time = response_time
            return response_answer

import os

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:/Users/Abdulrahman/Downloads/credentials.json'

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_model_selection_v2(project_id: str, model: str, audio_file: str, language: str) -> tuple:
    """Transcribe an audio file."""
    client = SpeechClient()

    with open(audio_file, "rb") as f:
        content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[language],
        model=model,
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        content=content,
    )

    response = client.recognize(request=request)

    highest_confidence = 0.0
    best_transcript = ""
    best_language = ""

    for result in response.results:
        transcript = result.alternatives[0].transcript
        confidence = result.alternatives[0].confidence
        detected_language = result.language_code
        print(f"Transcript: {transcript}, Confidence: {confidence}, Language: {detected_language}")
        
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_transcript = transcript
            best_language = detected_language

    return best_transcript, best_language, highest_confidence

def transcribe_audio_file(project_id, audio_file):
    # Try transcribing with English
    model = "latest_long"
    highest_confidence = 0.0
    best_transcript = ""
    best_language = ""

    try:
        transcript, language, confidence = transcribe_model_selection_v2(project_id, model, audio_file, "en-US")
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_transcript = transcript
            best_language = language
    except Exception as e:
        print(f"Error transcribing English: {e}")

    # Try transcribing with Arabic
    try:
        transcript, language, confidence = transcribe_model_selection_v2(project_id, model, audio_file, "ar-EG")
        if confidence > highest_confidence:
            highest_confidence = confidence
            best_transcript = transcript
            best_language = language
    except Exception as e:
        print(f"Error transcribing Arabic: {e}")

    return best_transcript, best_language


# Text to Speech Stage 

from google.cloud import texttospeech

def synthesize_speech(text, language_code, gender, output_file):
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, ssml_gender=gender
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is binary
    with open(output_file, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print(f'Audio content written to file "{output_file}"')



def transcribe_audio(file_path):
    model = st.session_state.get('model')
    if not model:
        st.error("The model is not loaded yet. Please load the model first.")
        return "", ""
    result = model.transcribe(file_path)
    if result['language'] == 'en':
        result['text'] = str.lower(result['text'])
        
    cleaned_text = clean_text(result['text'])
    return cleaned_text, result['language']

def record_and_save_audio(file_path):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Speak now...")
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
        st.info("Recording complete.")

        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        print("Audio saved as " + file_path)
        
# def record_and_save_audio(file_path, silence_threshold=100, silence_duration=3):
#     r = sr.Recognizer()
#     with sr.Microphone() as source:
#         st.info("Speak now...")
#         r.adjust_for_ambient_noise(source)
#         audio_data = []
#         last_speech_time = time.time()

#         while True:
#             try:
#                 audio = r.listen(source, timeout=5, phrase_time_limit=5)
#                 audio_data.append(audio.get_wav_data())

#                 # Analyze the audio energy level to detect speech
#                 energy = r.energy_threshold
#                 if energy > silence_threshold:
#                     last_speech_time = time.time()

#                 # If there is no speech detected for silence_duration seconds, stop recording
#                 if (time.time() - last_speech_time) > silence_duration:
#                     break

#             except sr.WaitTimeoutError:
#                 # No audio detected within the timeout period
#                 if (time.time() - last_speech_time) > silence_duration:
#                     break

#         st.info("Recording complete.")

#         # Save the recorded audio to file
#         with open(file_path, "wb") as f:
#             for chunk in audio_data:
#                 f.write(chunk)
#         print("Audio saved as " + file_path)



if __name__ == '__main__':
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    
    st.subheader('Question Answering Avatar')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key: ', type='password')
        pinecone_api_key = st.text_input('Pinecone API Key: ', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        if pinecone_api_key:
            os.environ['PINECONE_API_KEY'] = pinecone_api_key

        uploaded_file = st.file_uploader('Upload a file: ', type=['pdf'])
        chunk_size = st.number_input('Chunk size', min_value=100, max_value=2048, value=512)
        k = st.number_input('k', min_value=1, max_value=20, value=3)
        add_data = st.button('Add Data')
        
        if uploaded_file and add_data:
            with st.spinner('Reading, Chunking and Embedding the data'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size)
                st.write(f'Chunksize: {chunk_size}, Number of chunks: {len(chunks)}')
                delete_pinecone_index()
                vector_store = insert_or_fetch_embeddings('iti-indexx', chunks)
                st.session_state.vs = vector_store

                st.success('File is Uploaded, Chunked and Embedded successfully')

    if st.button('Use Microphone'):
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            audio_file_name = "user_input.wav"
            audio_file_path = os.path.join('./', audio_file_name)  # Relative path from where Streamlit is running
            record_and_save_audio(audio_file_path)
            project_id = "august-edge-427400-j2"
            q, language = transcribe_audio_file(project_id, audio_file_path)
            st.session_state.language = language
            st.write(f"Transcribed question: {q} (Language: {language})")
            answer = clean_answer(ask_context_aware(vector_store, q, k))
            st.session_state.answer = answer
            response_time = st.session_state.response_time
            st.text_area('Answer:', value=answer)

            # Update chat history
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q}\nA: {answer}\nResponse Time: {response_time:.2f} seconds'
            st.session_state.history = f'{value}\n{"-"*100}\n{st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
            
            # Text-to-Speech output


    q = st.text_input('Ask a question: ')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = clean_answer(ask_context_aware(vector_store, q, k))
            # answer = st.session_state.answer
            response_time = st.session_state.response_time
            st.text_area('Answer:', value=answer)

            # Update chat history
            if 'history' not in st.session_state:
                st.session_state.history = ''
            value = f'Q: {q}\nA: {answer}\nResponse Time: {response_time:.2f} seconds'
            st.session_state.history = f'{value}\n{"-"*100}\n{st.session_state.history}'
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
            
    if st.button('Generate Speech'):
        if 'vs' in st.session_state:
            answer = st.session_state.answer
            language = st.session_state.language
            tts_output_file = "D:\or_output.mp3"
            st.write("Generating speech output...")
            synthesize_speech(text=answer, language_code=language, gender=texttospeech.SsmlVoiceGender.NEUTRAL, output_file=tts_output_file)
            st.write("Speech output generated.")

            if os.path.exists(tts_output_file):
                st.audio(tts_output_file, format="audio/mp3")
                
            if 'history' not in st.session_state:
                st.session_state.history = ''
            h = st.session_state.history
            st.text_area(label='Chat History', value=h, key='history', height=400)
            

# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# # Example data, replace with your actual data extracted from the PDF
# data_samples = {
#     'question': [
#         'What is the main mission of the Information Technology Institute (ITI)?',
#         'When was ITI established and when did it become an affiliate of MCIT?  ',
#         'How does ITI sustain an 85% employment rate for its graduates? ',
#         ''
        
#     ],
#     'answer': [
#         'The director of the technology institute is Dr. Ahmed Mansour.',
#         'The technology institute offers courses in Computer Science, Information Technology, and Cyber Security.',
#     ],
#     'contexts': [
#         ['The technology institute, led by Dr. Ahmed Mansour, focuses on...',
#          'Under the direction of Dr. Ahmed Mansour, the institute has...'],
#         ['The institute offers a variety of courses including Computer Science...',
#          'Courses at the institute include Information Technology, Cyber Security...'],
#     ],
#     'ground_truth': [
#         'The director of the technology institute is Dr. Ahmed Mansour.',
#         'Courses offered include Computer Science, Information Technology, and Cyber Security.'
#     ]
# }

# # Convert the dictionary to a Dataset object
# dataset = Dataset.from_dict(data_samples)

# # Evaluate the dataset using the specified metrics
# score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

# # Convert the scores to a DataFrame for easier viewing
# scores_df = score.to_pandas()
# print(scores_df)



# import pdfplumber

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"
#     return text

# # Path to your PDF file
# pdf_path = "D:\intake44\GP\complete_pipeline\data_set_final_version.pdf"
# document_text = extract_text_from_pdf(pdf_path)

# import openai


# def generate_qa_pairs(text, max_pairs=10):
#     prompt = (f"The following is a passage extracted from a document:\n\n{text}\n\n"
#               "Based on this passage, generate a series of question-answer pairs. The questions should be clear and "
#               "specific, and the answers should be concise and accurate. Each question-answer pair should be labeled "
#               "with 'Q:' for the question and 'A:' for the answer. Limit the response to {} pairs.".format(max_pairs))
    
#     response = openai.Completion.create(
#         model="text-davinci-003", # Use the most appropriate GPT-4 model available to you
#         prompt=prompt,
#         max_tokens=1500,
#         n=1,
#         stop=None,
#         temperature=0.7,
#     )

#     return response['choices'][0]['text']

# # Generate question-answer pairs
# qa_pairs_text = generate_qa_pairs(document_text)
# print(qa_pairs_text)


# def parse_qa_pairs(qa_text):
#     pairs = qa_text.strip().split('\n\n')
#     questions = []
#     answers = []
#     for pair in pairs:
#         if 'Q:' in pair and 'A:' in pair:
#             q, a = pair.split('A:', 1)
#             q = q.replace('Q:', '').strip()
#             a = a.strip()
#             questions.append(q)
#             answers.append(a)
#     return questions, answers

# # Parse the output
# questions, answers = parse_qa_pairs(qa_pairs_text)

# # Display the results
# for q, a in zip(questions, answers):
#     print(f"Question: {q}")
#     print(f"Answer: {a}")
#     print("-----------")

# from datasets import Dataset

# # Assuming the whole document text is used as context for simplicity
# data_samples = {
#     'question': questions,
#     'answer': answers,
#     'contexts': [[document_text] for _ in questions],  # The context for each question
#     'ground_truth': answers  # Using answers as ground truth for simplicity
# }

# # Convert the dictionary to a Dataset object
# dataset = Dataset.from_dict(data_samples)

# # Evaluate the dataset using the specified metrics
# score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

# # Convert the scores to a DataFrame for easier viewing
# scores_df = score.to_pandas()
# print(scores_df)



