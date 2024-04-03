__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import uuid
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from conversation_template import css, bot_template, user_template

OPENAI_API_KEY = st.secrets["openai_api_key"]

# Function for creating relevant context
def get_text_snippet(text, question, window_size=500):
    # Load a pre-trained QA model and tokenizer
    qa_pipeline = pipeline("question-answering", model="distilbert/distilbert-base-cased-distilled-squad")

    # Prepare the inputs for the model
    inputs = {
        "question": question,
        "context": text
    }

    # Get the answer from the model
    result = qa_pipeline(inputs)

    # Check if the model found an answer within the context
    if not result['answer']:
        return "The model could not find an answer in the context provided."

    # Find the end positions of the answer in the context
    end_position = result['end']
    print('Start position : ',result['start'])
    print('End position : ', end_position)

    # Calculate the end snippet position, expanding around the answer based on the window size
    end_snippet = min(len(text), end_position + window_size)

    # Set the start of the snippet to the beginning of the text
    start_snippet = 0

    # Extract and return the snippet containing the answer
    snippet = text[start_snippet:end_snippet]
    print("Length of Given Context : ", len(text))
    print("Length of Initial Generated Relevant Text : ", len(snippet))

    # checking if given context and generated context length is same or not
    if len(text) == len(snippet):
        start_position = result['start']
        end_position = result['end']
        
        # Adjust the start and end snippet positions to center around the answer
        snippet_length = window_size // 2  # Half before, half after the answer
        start_snippet = max(0, start_position - snippet_length)
        end_snippet = min(len(text), end_position + snippet_length)

        # Ensure the snippet doesn't start in the middle of a word
        if start_snippet > 0 and text[start_snippet - 1].isalnum():
            start_snippet = text.rfind(" ", 0, start_snippet) + 1

        snippet = text[start_snippet:end_snippet]
        print('Length of final Snippet : ', len(snippet))
        return snippet                                  
    else:
        print('Length of final Snippet : ', len(snippet))
        return snippet


# Function for getting the final answer
def get_answer_from_context(context, question):
    # Assuming the creation of documents from the context is stateless and can be repeated safely
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents([context])
    
    # Generate unique IDs for each document
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in texts]
    unique_ids = list(set(ids))

    # Filter for unique documents
    seen_ids = set()
    unique_docs = [doc for doc, id in zip(texts, ids) if id not in seen_ids and (seen_ids.add(id) or True)]

    # Reinitialize embeddings and Chroma database for each invocation to avoid state retention
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(documents=unique_docs, embedding=embeddings, ids=unique_ids)

    # Initialize the QA system fresh for each call
    eqa = VectorDBQA.from_chain_type(
        llm=OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0),
        chain_type='refine',
        vectorstore=docsearch,
        return_source_documents=True
    )
    
    # Attempt to get the answer using the freshly initialized system
    answer = eqa.invoke(question)
    docsearch.delete_collection()
    print("QA result", answer['result'])
    return answer


# Streamlit UI
def display_text_eqa_ui():
    # Reset relevant session state variables
    st.session_state['text_specific_state'] = None  # Reset or clear as needed
    

    st.header("Extractive Question Answering System for Text Context")

    question_input = st.text_area("Enter Your Question here", key="question_input")
    context_input = st.text_area("Enter Your Context here", height=300, key="context_input")

    generate_relevant_text_button = st.button("Get Relevant Text")

    # Use local variables to hold intermediate results instead of session state
    if generate_relevant_text_button:
        # Reset or clear previous values stored in session state before processing
        #st.session_state['output_text'] = ""
        #st.session_state['final_answer'] = ""
        #del st.session_state['output_text']
        #del st.session_state['final_answer']
        st.session_state['output_text'] = None
        st.session_state['final_answer'] = None

        relevant_text = get_text_snippet(context_input, question_input)
        if "The model could not find an answer in the context provided." not in relevant_text:
            print("Relevant Context : ", relevant_text)
            print("Question : ", question_input)
            result = get_answer_from_context(relevant_text, question_input)
            final_answer = result['result'].strip()  
        else:
            relevant_text = ""
            final_answer = ""

        # Use local variables to update session state directly
        st.session_state['output_text'] = relevant_text
        st.session_state['final_answer'] = final_answer

    # Display the relevant text
    if 'output_text' in st.session_state and st.session_state['output_text']:
        st.text_area("Relevant Text", value=st.session_state['output_text'], height=250, disabled=True, key="output_result")

    # Display the final answer
    if 'final_answer' in st.session_state and st.session_state['final_answer']:
        st.text_area("Final Answer:", value=st.session_state['final_answer'], height=300, disabled=True, key="final_answer_display")


def get_pdf_texts(pdf_documents):
    # initializing an empty variable for storing pdf documents
    raw_text = ""

    # iterating through pdf documents
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    
    return raw_text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )

    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def create_conversation_chain(vectore_store):
    llm = ChatOpenAI(openai_api_key = OPENAI_API_KEY)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectore_store.as_retriever(),
        memory = memory
    )

    return conversation_chain

def handle_user_input(question):
    # Ensure there's a conversation chain to handle the question
    if st.session_state.conversation is not None:
        # Process the question through the conversation chain
        response = st.session_state.conversation({"question": question})
        st.session_state.chat_history = response['chat_history']

        # Display the conversation history
        for i, message in enumerate(st.session_state.chat_history):
            # Alternating between user and bot messages for display
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", "User : "+ message.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", "LLM: "+ message.content), unsafe_allow_html=True)
    else:
        # Prompt the user to process documents first if the conversation chain isn't ready
        st.error("Please process your PDF documents first to initialize the conversation.")          


def generate_pdf_eqa_ui():
    # Reset relevant session state variables
    st.session_state['pdf_specific_state'] = None  # Reset or clear as needed
    
    st.write(css, unsafe_allow_html=True)
    st.header("Extractive Question Answering System for PDF Documents")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    pdf_docs = st.file_uploader("Upload your file here and click `Process`", accept_multiple_files=True)
    process_button = st.button("Process")

    if process_button:
        if not pdf_docs:  # Check if the user pressed the process button without uploading documents
            st.error("Please upload document/s before pressing 'Process'.")
        else:
            with st.spinner("Processing"):
                # get pdf texts
                pdf_texts = get_pdf_texts(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(pdf_texts)

                # create vector store embeddings
                vectore_store = get_vector_store(text_chunks)
                
                # create conversation chain
                st.session_state.conversation = create_conversation_chain(vectore_store)

    user_question = st.text_input("Ask a question about uploaded document")
    ask_question_button = st.button("Ask")

    if ask_question_button:
        if not user_question:  # Check if the user pressed the ask button without entering a question
            st.error("Please enter a question before pressing 'Ask'.")
        if st.session_state.conversation is None:  # Checks if the document has been processed
            st.error("""Please follow the steps to ask question:
                        1. Upload Document/s
                        2. Process the document/s
                        3. Ask Question
                        4. Then press Ask button to get answer"""
                    )
        else:
            handle_user_input(user_question)

def main():
    st.set_page_config(page_title="Extractive Question Answering System", layout='centered')
    st.title("Extractive Question Answering Using LLM")

    if 'current_page' not in st.session_state:
        st.session_state.current_page = None

    col1, col2 = st.columns(2)
    with col1:
        if st.button("PDF EQA"):
            #Reset state specific to Text EQA here
            st.session_state['output_text'] = None
            st.session_state['final_answer'] = None
            # Reset PDF specific state variables
            st.session_state['conversation'] = None
            st.session_state['chat_history'] = None
            # set the current page to PDF EQA mode
            st.session_state.current_page = "pdf"
    with col2:
        if st.button("Text EQA"):
            # Reset state specific to PDF EQA here if necessary
            st.session_state.current_page = "text"

    if st.session_state.current_page == "pdf":
        generate_pdf_eqa_ui()  # Function from pdf_eqa.py
    elif st.session_state.current_page == "text":
        display_text_eqa_ui()  # Function from text_eqa.py

if __name__ == '__main__':
    main()
