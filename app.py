import os
import uuid
import streamlit as st
import pandas as pd
import pdfplumber
import chromadb
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import re

# ---------------- Config ----------------
st.set_page_config(
    page_title="MR Analyst",
    page_icon="📊",
    layout="wide"
)

# ---------------- Slate Gray Theme ----------------
st.markdown("""
<style>

/* ================= Background ================= */
.stApp{
    background:linear-gradient(180deg,#F8FAFC 0%,#EEF2F7 100%);
}

/* Hide Streamlit branding */
header[data-testid="stHeader"]{background:transparent;}
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

/* ================= Typography ================= */
h1,h2,h3{
    color:#0F172A !important;
    font-weight:700;
}

/* Normal text */
.stMarkdown p,
.stMarkdown div,
label,
.stCaption,
.stText{
    color:#0F172A !important;
}

hr{
    border:0;
    border-top:1px solid #CBD5E1;
}

/* ================= Buttons ================= */
.stButton>button{
    width:100%;
    height:62px;
    border:1px solid #D6DEE8;
    border-radius:18px;
    background:white;
    color:#000000 !important;
    font-weight:700;
    font-size:17px;
    box-shadow:0 3px 10px rgba(15,23,42,.08);
    transition:.25s;
}

.stButton>button:hover{
    background:#EEF4FF;
    border-color:#4F6BED;
    color:#000000 !important;
    transform:translateY(-1px);
}

/* Spacing between vertical buttons */
.button-space{
    margin-top:12px;
}

/* ================= Upload ================= */
[data-testid="stFileUploader"]{
    background:white;
    border:2px dashed #CBD5E1;
    border-radius:22px;
    padding:28px;
}

[data-testid="stFileUploader"] button{
    background:#334155 !important;
    color:white !important;
    border-radius:12px;
}

/* ================= Text Input ================= */
.stTextInput input{
    background:#F1F5F9;
    border:1px solid #CBD5E1;
    border-radius:12px;
    color:#111827 !important;
}

/* ================= Expander ================= */
details{
    background:white;
    border:1px solid #E2E8F0;
    border-radius:14px;
    padding:12px;
}

/* =====================================================
   ALERT CARDS - BLACK TEXT FIX (Latest Streamlit)
===================================================== */

div[data-testid="stAlert"]{
    border-radius:16px !important;
    padding:14px 16px !important;
    box-shadow:0 2px 8px rgba(15,23,42,.08);
}

/* Force every text element inside alerts to BLACK */
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] *,
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] div,
div[data-testid="stAlert"] strong{
    color:#000000 !important;
    fill:#000000 !important;
    stroke:none !important;
    font-weight:600 !important;
}

/* Keep icons colored */
div[data-testid="stAlert"][kind="warning"] svg{
    color:#D4A000 !important;
}

div[data-testid="stAlert"][kind="success"] svg{
    color:#2E8B57 !important;
}

div[data-testid="stAlert"][kind="info"] svg{
    color:#2563EB !important;
}

div[data-testid="stAlert"][kind="error"] svg{
    color:#DC2626 !important;
}

/* Warning */
div[data-testid="stAlert"][kind="warning"]{
    background:#FFF4CC !important;
    border-left:5px solid #D4A000 !important;
}

/* Success */
div[data-testid="stAlert"][kind="success"]{
    background:#DFF6E8 !important;
    border-left:5px solid #2E8B57 !important;
}

/* Info */
div[data-testid="stAlert"][kind="info"]{
    background:#E8F1FF !important;
    border-left:5px solid #3B82F6 !important;
}

/* Error */
div[data-testid="stAlert"][kind="error"]{
    background:#FFE6E6 !important;
    border-left:5px solid #DC2626 !important;
}

/* ================= SQL Code Block ================= */
.stCode{
    border-radius:18px !important;
    border:1px solid #334155 !important;
    overflow:hidden;
}

.stCode pre{
    background:#0B1220 !important;
    font-size:17px !important;
    line-height:1.8 !important;
    padding:22px !important;
}

/* Preserve SQL syntax highlighting */
.stCode code,
.stCode span{
    color:inherit !important;
    font-family:"Cascadia Code","Consolas","Courier New",monospace !important;
}

/* Copy button */
.stCode button{
    background:#1E293B !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- MR Analyst Header ----------------
left, right = st.columns([5, 2])

with left:
    st.markdown("## 📊 MR Analyst")
    st.caption("AI Platform for Data Analysts")

with right:
    st.markdown("""
    <div style="
        background:white;
        border:1px solid #E2E8F0;
        border-radius:18px;
        padding:16px;
        text-align:center;
        box-shadow:0 4px 12px rgba(15,23,42,.08);
    ">
        <div style="font-size:24px;">✨</div>
        <div style="font-weight:700;color:#334155;">Turning Data into</div>
        <div style="font-size:18px;font-weight:800;color:#2563EB;">Decisions</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)



# ---------------- Upload ----------------
st.markdown("### Upload Files")

files = st.file_uploader(
    "",
    type=["csv","xlsx","xls","pdf","txt"],
    accept_multiple_files=True
)

if files:
    st.session_state.files = files

# ---------------- Manual Workflow ----------------
st.markdown("<div class='button-space'></div>", unsafe_allow_html=True)

extract_clicked = st.button(
    "📄  Extract Data",
    use_container_width=True
)

st.markdown("<div class='button-space'></div>", unsafe_allow_html=True)

chunk_clicked = st.button(
    "📚  Create Chunks & Embeddings",
    use_container_width=True
)

st.markdown("<div class='button-space'></div>", unsafe_allow_html=True)

store_clicked = st.button(
    "🗄️  Store in ChromaDB",
    use_container_width=True
)

# Extract
if extract_clicked:

    if "files" not in st.session_state:
        st.warning("Upload files first.")

    else:

        docs=[]

        with st.spinner("Extracting..."):

            for f in st.session_state.files:

                docs.append({
                    "filename":f.name,
                    "content":extract(f)
                })

        st.session_state.docs=docs
        st.success(f"Extracted {len(docs)} file(s).")

# Chunks
if chunk_clicked:

    if "docs" not in st.session_state:
        st.warning("Extract data first.")

    else:

        chunks=[]
        embeds=[]

        with st.spinner("Creating embeddings..."):

            for doc in st.session_state.docs:

                c=chunk(doc["content"])
                e=model.encode(c)

                chunks.extend(
                    [{"filename":doc["filename"],"content":x} for x in c]
                )

                embeds.extend(e)

        st.session_state.chunks=chunks
        st.session_state.embeds=embeds

        st.success("Chunks & Embeddings created.")
        st.info(
            f"Chunk Size: {CHUNK_SIZE} | Overlap: {OVERLAP} | Total Chunks: {len(chunks)}"
        )

# Store
if store_clicked:

    if "chunks" not in st.session_state:
        st.warning("Create chunks first.")

    else:

        with st.spinner("Storing in ChromaDB..."):

            collection.upsert(

                documents=[c["content"] for c in st.session_state.chunks],

                embeddings=[e.tolist() for e in st.session_state.embeds],

                metadatas=[
                    {"filename":c["filename"]}
                    for c in st.session_state.chunks
                ],

                ids=[
                    f"{c['filename']}_{uuid.uuid4().hex}"
                    for c in st.session_state.chunks
                ]
            )

        st.success(f"Stored {len(st.session_state.chunks)} chunks.")

# ---------------- Quick Workflow ----------------
st.divider()

st.markdown("""
<h2>⚡ Quick Workflow</h2>
<p style='color:#64748B;margin-top:-10px;'>
Run the complete pipeline with one click.
</p>
""", unsafe_allow_html=True)

if st.button("▶ Automate Everything"):

    if "files" not in st.session_state:
        st.warning("Upload files first.")

    else:

        progress=st.progress(0)
        status=st.empty()

        docs=[]
        chunks=[]
        embeds=[]

        status.info("Step 1/3 • Extracting")

        for f in st.session_state.files:

            docs.append({
                "filename":f.name,
                "content":extract(f)
            })

        progress.progress(30)

        status.info("Step 2/3 • Creating Embeddings")

        for doc in docs:

            c=chunk(doc["content"])
            e=model.encode(c)

            chunks.extend(
                [{"filename":doc["filename"],"content":x} for x in c]
            )

            embeds.extend(e)

        progress.progress(70)

        status.info("Step 3/3 • Storing")

        collection.upsert(

            documents=[c["content"] for c in chunks],

            embeddings=[e.tolist() for e in embeds],

            metadatas=[
                {"filename":c["filename"]}
                for c in chunks
            ],

            ids=[
                f"{c['filename']}_{uuid.uuid4().hex}"
                for c in chunks
            ]
        )

        progress.progress(100)

        st.session_state.docs=docs
        st.session_state.chunks=chunks
        st.session_state.embeds=embeds

        status.success("Completed!")
        st.success(f"Stored {len(chunks)} chunks.")

# ---------------- Ask AI ----------------
st.divider()

st.markdown("""
<h2>Ask AI</h2>
<p style='color:#64748B;margin-top:-10px;'>
Get answers directly from your uploaded documents.
</p>
""", unsafe_allow_html=True)

question=st.text_input("Question")
top_n=st.slider("Top Chunks",1,10,3)

col1,col2=st.columns(2)

with col1:
    retrieve=st.button("🔍 Retrieve Chunks")

with col2:
    ask=st.button("💬 Ask AI")

# Retrieve
if retrieve:

    if collection.count()==0:
        st.warning("No documents stored.")

    elif not question.strip():
        st.warning("Enter a question.")

    else:

        result=collection.query(
            query_embeddings=[model.encode(question).tolist()],
            n_results=top_n
        )

        st.session_state.result=result

        st.success(f"Retrieved {top_n} chunks.")

# View chunks
if "result" in st.session_state:

    with st.expander("View Retrieved Chunks"):

        docs=st.session_state.result["documents"][0]
        meta=st.session_state.result["metadatas"][0]

        for i,text in enumerate(docs,1):

            st.markdown(f"**Chunk {i}**")
            st.write(text)
            st.caption(meta[i-1]["filename"])

# Ask AI
if ask:

    if "result" not in st.session_state:
        st.warning("Retrieve chunks first.")

    else:

        context="\n\n".join(
            st.session_state.result["documents"][0]
        )

        with st.spinner("Thinking..."):

            reply=client.chat.completions.create(

                model="openai/gpt-oss-20b",

                temperature=0.2,

                messages=[
                    {
                        "role":"system",
                        "content":"Answer ONLY using the provided context."
                    },
                    {
                        "role":"user",
                        "content":f"""
Context:
{context}

Question:
{question}
"""
                    }
                ]
            )

        st.subheader("Answer")
        st.markdown(reply.choices[0].message.content)

# ---------------- SQL Generator ----------------
st.divider()

st.markdown("""
<h2>SQL Generator</h2>
<p style='color:#64748B;margin-top:-10px;'>
Generate MySQL queries from your uploaded CSV.
</p>
""", unsafe_allow_html=True)

if st.button("🗃️ Generate SQL"):

    csvs=[
        f for f in st.session_state.get("files",[])
        if f.name.lower().endswith(".csv")
    ]

    if len(csvs)!=1:
        st.warning("Upload exactly one CSV.")

    elif not question.strip():
        st.warning("Enter your SQL requirement.")

    else:

        csv_file=csvs[0]
        csv_file.seek(0)

        df=pd.read_csv(csv_file)

        schema="\n".join(
            f"{c}: {t}"
            for c,t in df.dtypes.astype(str).items()
        )

        sample=df.head().to_dict("records")

        response=client.chat.completions.create(

            model="openai/gpt-oss-20b",

            temperature=0.1,

            messages=[
                {
                    "role":"system",
                    "content":"""
You are an expert MySQL developer.

Generate ONLY MySQL.
Return SQL inside a sql code block.
Use only available columns.
"""
                },
                {
                    "role":"user",
                    "content":f"""
Schema:
{schema}

Sample Data:
{sample}

Requirement:
{question}
"""
                }
            ]
        )

        

        st.subheader("Generated SQL")

        sql_text = response.choices[0].message.content

        # Extract SQL from ```sql ... ``` if present
        match = re.search(r"```sql\n(.*?)```", sql_text, re.DOTALL)

        if match:
            st.code(match.group(1).strip(), language="sql")
        else:
            st.code(sql_text.strip(), language="sql")

# ---------------- Footer ----------------
st.divider()

st.markdown("""
<div style="display:flex;justify-content:space-between;color:#64748B;font-size:15px;padding-bottom:10px;">
    <span><b>MR Analyst</b> | AI Platform for Data Analysts</span>
    <span>Built with ❤️ by Tanmay</span>
</div>
""", unsafe_allow_html=True)