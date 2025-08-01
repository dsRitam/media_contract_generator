import streamlit as st
from datetime import datetime
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import re
import fitz

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file!")
    st.stop()

# <-------------- LOADING CONTRACT PDF ----------->
@st.cache_data
def extract_pdf_text(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if not text.strip():
            st.error("No text extracted. Ensure the PDF contains selectable text.")
            return None
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

# <---------- RAG Setup Function ---------->
@st.cache_resource
def get_retriever_and_prompt(_contract_doc):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([_contract_doc])
    if not chunks:
        st.error("Could not split the document into chunks.")
        return None, None
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # This prompt is now used for both full generation and updates.
    contract_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a legal expert AI. Your task is to draft or modify a contract using the provided context as a template for structure, tone, and legal language.
        The final output must be in **Markdown format**.

        - For DRAFTING: Fill in the specific details provided by the user.
        - For MODIFICATIONS: Update ONLY the requested sections or clauses. When modifying a section, return the ENTIRE updated section text in Markdown format, starting with its '##' heading.

        Context from Template:
        {context}

        User Request:
        {question}

        Drafted or Modified Contract Section/Point (in Markdown):
        """
    )
    return retriever, contract_prompt

# <---------- DELETED The static markdown_template variable ----------->

# <--------- Fetching User Input ------------>
def collect_inputs():
    with st.sidebar:
        st.header("Contract Details")
        st.subheader("1. Upload Contract Template")
        uploaded_file = st.file_uploader("Upload your template PDF", type="pdf")
        st.subheader("2. Producer Details")
        producer_details = {
            "producer_name": st.text_input("Your Company Name", "Concept Cube"),
            "producer_address": st.text_area("Your Company Address", "Concept Building, Prince Park, Kalkikapur Road, Kolkata, 700099, West Bengal, India"),
            "producer_email": st.text_input("Your Company Email", "hello@conceptcube.in"),
            "producer_phone": st.text_input("Your Company Phone", "+91 8900707330"),
            "producer_rep_name": st.text_input("Your Representative's Name", "Authorized Signatory"),
            "producer_rep_title": st.text_input("Your Representative's Title", "Director"),
        }
        st.subheader("3. Client & Project Details")
        effective_date_obj = st.date_input("Effective Date", datetime.now())
        client_details = {
            "client_name": st.text_input("Client Name"),
            "client_legal_status": st.text_input("Client Legal Status (e.g., company)"),
            "client_address": st.text_area("Client Address"),
            "client_rep_name": st.text_input("Client Representative's Name"),
            "client_rep_title": st.text_input("Client Representative's Title"),
            "effective_date": effective_date_obj.strftime("%d %B %Y"),
            "project_description": st.text_area("Project Description"),
            "deliverables": st.text_area("List of Deliverables"),
            "timeline": st.text_area("Project Timeline"),
            "total_fee": st.text_input("Total Project Fee (e.g., INR 500,000)"),
            "payment_schedule": st.text_area("Payment Schedule", "50% advance, 50% on completion."),
        }
    all_details = {**producer_details, **client_details}
    required_fields = ["client_name", "client_legal_status", "client_address", "client_rep_name", "client_rep_title", "project_description", "total_fee"]
    missing_fields = [key.replace('_', ' ').title() for key, value in all_details.items() if key in required_fields and not value]
    return uploaded_file, all_details, missing_fields

# <------- NEW: Function to generate the entire contract dynamically ---->
def generate_full_contract(chain, details):
    """Generates the full contract by providing all details to the LLM."""
    
    # Create a clean string of all user-provided details
    details_string = "\n".join([f"- {key.replace('_', ' ').title()}: {value}" for key, value in details.items() if value])

    query = f"""
    Draft a complete contract from beginning to end in Markdown format.
    Use the provided context as the primary template for structure, clauses, and legal language.
    Integrate the following specific details into the document, replacing any placeholders or examples from the context:

    {details_string}

    The final output must be a single, complete, and well-formatted Markdown document.
    """
    return chain.invoke(query)

# <--------- Update specific parts (No changes needed here) ---------->
def update_contract(retriever, prompt, llm_with_stop, current_contract, section, user_prompt):
    section_pattern = re.compile(rf"(^##\s*{re.escape(section)}.*?)(?=^##\s|\Z)", re.DOTALL | re.MULTILINE)
    match = section_pattern.search(current_contract)
    if not match:
        st.error(f"Could not find section '{section}' to update.")
        return current_contract

    query = f"""
    Based on the user's request, update the following contract section.
    Return ONLY the complete, updated text for this single section, starting with its '##' heading.
    ABSOLUTELY DO NOT include any subsequent sections or any text that comes after this section.

    Original Section Text to Update:
    ---
    {match.group(1)}
    ---

    User Request: "{user_prompt}"
    """
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    update_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_with_stop
        | StrOutputParser()
    )
    
    updated_section_text = update_chain.invoke(query)
    updated_contract = section_pattern.sub(updated_section_text.strip(), current_contract, count=1)
    
    return updated_contract

# <--------------- Streamlit UI -------------->
def main():
    st.set_page_config(layout="wide")
    st.title("📄 Dynamic Contract Generator")

    if "contract" not in st.session_state:
        st.session_state.contract = None

    uploaded_file, details, missing_fields = collect_inputs()

    if not uploaded_file:
        st.info("Please upload a contract template in the sidebar to begin.")
        st.stop()
    
    contract_text = extract_pdf_text(uploaded_file)
    if not contract_text:
        st.stop()

    contract_doc = Document(page_content=contract_text, metadata={"source": uploaded_file.name})
    retriever, contract_prompt = get_retriever_and_prompt(contract_doc)

    if not retriever:
        st.stop()
    
    # --- LLM and Chain Definitions ---
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1
    )
    llm_with_stop = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        stop=["\n## "]
    )

    # General purpose RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | contract_prompt
        | llm
        | StrOutputParser()
    )

    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Generation & Modification")
        # --- CHANGED: Logic for generating the contract ---
        if st.button("Generate Full Contract", disabled=bool(missing_fields)):
            with st.spinner("Generating dynamic contract... This may take a moment."):
                # Call the new function to generate the entire contract
                full_contract = generate_full_contract(rag_chain, details)
                st.session_state.contract = full_contract
                st.success("Contract generated successfully!")
        elif missing_fields:
                st.warning(f"Please fill in required fields in the sidebar: {', '.join(missing_fields)}")
        # --- END CHANGE ---

        if st.session_state.contract:
            st.subheader("Modify Contract Section")
            section_titles = re.findall(r"^##\s*(.*)", st.session_state.contract, re.MULTILINE)
            section_to_update = st.selectbox("Select Section to Update", section_titles, index=len(section_titles)-1 if section_titles else 0)
            update_prompt = st.text_area("Describe the changes:", placeholder="e.g., 'Change the total fee to $750,000 and the payment schedule to 60% advance.'")

            if st.button("Update Contract Section") and update_prompt and section_to_update:
                with st.spinner("Applying updates..."):
                    updated_contract = update_contract(retriever, contract_prompt, llm_with_stop, st.session_state.contract, section_to_update, update_prompt)
                    st.session_state.contract = updated_contract
                    st.success(f"Section '{section_to_update}' updated!")

    with col2:
        st.header("Live Contract Preview")
        if st.session_state.contract:
            st.markdown(st.session_state.contract)
            st.download_button(
                label="Download .md File",
                data=st.session_state.contract,
                file_name=f"Contract_{details.get('client_name', 'Client').replace(' ', '_')}.md",
                mime="text/markdown"
            )
        else:
            st.info("Your generated contract will be displayed here.")

if __name__ == "__main__":
    main()