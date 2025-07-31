import streamlit as st
from datetime import datetime
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
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
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found in .env file!")
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

# <---------- Markdown Template ----------->
markdown_template = """
# MEDIA PRODUCTION SERVICES AGREEMENT

**Company:**
{producer_name}
{producer_address}
Email: {producer_email}
Phone: {producer_phone}

---

This Media Production Services Agreement (the "Agreement") is entered into as of this **{effective_date}** (the "Effective Date").

**BETWEEN:**

**{producer_name}**, a company incorporated under the laws of India, with its principal place of business at {producer_address} (hereinafter referred to as the "Producer").

**AND:**

**{client_name}**, a {client_legal_status}, with its principal address at {client_address} (hereinafter referred to as the "Client").

The Producer and the Client are hereinafter individually referred to as a "Party" and collectively as the "Parties".

---

## RECITALS

**WHEREAS**, the Producer is in the business of providing professional media production services.
**WHEREAS**, the Client desires to engage the Producer to provide such services for a specific project.
**NOW, THEREFORE**, in consideration of the mutual covenants and promises contained herein, the Parties agree as follows:

{contract_body}

---

## IN WITNESS WHEREOF

The Parties have executed this Agreement as of the Effective Date.

**FOR {producer_name} ("PRODUCER")**

____________________________
**Name:** {producer_rep_name}
**Title:** {producer_rep_title}
**Date:** {effective_date}


**FOR {client_name} ("CLIENT")**

____________________________
**Name:** {client_rep_name}
**Title:** {client_rep_title}
**Date:** {effective_date}
"""

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

# <------- Function to generate contract ---->
def generate_contract_body(chain, details):
    query = f"""
    Draft the full contract body in Markdown format, starting from '## ARTICLE 1: DEFINITIONS' through to the final '## SCHEDULE A'.
    Incorporate the following project details:
    - Project Description: {details['project_description']}
    - Deliverables: {details['deliverables']}
    - Timeline: {details['timeline']}
    - Total Fee: {details['total_fee']}
    - Payment Schedule: {details['payment_schedule']}
    Ensure all articles from the template (e.g., Scope, Fees, IP, etc.) are included and populated correctly using Markdown headings.
    """
    return chain.invoke(query)

# <--------- Update specific parts ---------->
# FIX: The entire update logic is revised for robustness.
def update_contract(retriever, prompt, llm_with_stop, current_contract, section, user_prompt):
    """Updates a section of the contract using a dedicated LLM with a stop sequence."""
    section_pattern = re.compile(rf"(^##\s*{re.escape(section)}.*?)(?=^##\s|\Z)", re.DOTALL | re.MULTILINE)
    match = section_pattern.search(current_contract)
    if not match:
        st.error(f"Could not find section '{section}' to update.")
        return current_contract

    # FIX: A more directive prompt for the update task.
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
    
    # Build a temporary chain for this specific call
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    update_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_with_stop  # Using the LLM with the stop sequence
        | StrOutputParser()
    )
    
    updated_section_text = update_chain.invoke(query)
    
    # Replace the old section with the cleaned new one.
    # strip() removes any residual whitespace the LLM might add.
    updated_contract = section_pattern.sub(updated_section_text.strip(), current_contract, count=1)
    
    return updated_contract

# <--------------- Streamlit UI -------------->
def main():
    st.set_page_config(layout="wide")
    st.title("📄 Media Production Contract Generator")

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
    # General purpose LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    
    # FIX: Specialized LLM for updates with a stop sequence to prevent over-generation
    llm_with_stop = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1, stop=["\n## "])

    # General purpose RAG chain for full generation
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | contract_prompt
        | llm
        | StrOutputParser()
    )

    # FIX: Adjust column widths for a better preview experience
    col1, col2 = st.columns([2, 3])

    with col1:
        st.header("Generation & Modification")
        if st.button("Generate Full Contract", disabled=bool(missing_fields)):
            with st.spinner("Generating contract..."):
                contract_body = generate_contract_body(rag_chain, details)
                full_contract = markdown_template.format(contract_body=contract_body, **details)
                st.session_state.contract = full_contract
                st.success("Contract generated successfully!")
        elif missing_fields:
             st.warning(f"Please fill in required fields in the sidebar: {', '.join(missing_fields)}")

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