from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.paths import BASE_DIR, UPLOAD_FOLDER
from utils.database import (
    count_documents,
    create_note,
    create_tables,
    delete_document,
    delete_note,
    get_documents,
    get_notes,
    save_document,
)
from utils.file_handler import extract_text
from utils.text_tools import chunk_text
from utils.embeddings import create_embeddings, create_query_embedding
from utils.vector_store import add_documents, search_documents, vector_store_info
from utils.tfidf_search import tfidf_search
from utils.dataset_search import search_dataset
from utils.dataset_qa import answer_dataset_question
from utils.rag import ask_rag
from utils.summarizer import (
    extract_keywords,
    generate_abstract,
    generate_conclusion,
    generate_limitations,
    generate_main_findings,
    generate_summary,
)
from utils.translator import supported_languages, translate_text
from utils.research_tools import (
    compare_papers,
    compare_similarity,
    generate_citation,
    generate_literature_review,
    generate_research_questions,
)
from utils.dataset_analyzer import (
    compare_numeric_by_group,
    get_dataset_overview,
    load_dataset,
)
from utils.voice import text_to_speech
from utils.report_generator import export_docx, export_pdf, export_txt
from utils.ui_components import hero, info_card, load_css, result_panel, section_header, status_pill


st.set_page_config(
    page_title="Professional AI Research Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css(BASE_DIR / "assets" / "css" / "main.css")
create_tables()


SESSION_DEFAULTS = {
    "chat_history": [],
    "summary": "",
    "abstract": "",
    "keywords": [],
    "questions": "",
    "translation": "",
    "citation": "",
    "literature_review": "",
    "paper_comparison": "",
    "main_findings": "",
    "limitations": "",
    "conclusion": "",
    "last_report_path": "",
}

for key, default in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, default)


TEXT_TYPES = {".pdf", ".docx", ".txt", ".pptx"}
DATASET_TYPES = {".csv", ".xlsx", ".xls"}


def existing_names() -> list[str]:
    return [
        row[1]
        for row in get_documents()
        if (UPLOAD_FOLDER / row[1]).exists()
    ]


def read_document(filename: str) -> str:
    return extract_text(str(UPLOAD_FOLDER / filename))


def text_documents() -> list[str]:
    return [
        name for name in existing_names()
        if Path(name).suffix.lower() in TEXT_TYPES
    ]


def dataset_documents() -> list[str]:
    return [
        name for name in existing_names()
        if Path(name).suffix.lower() in DATASET_TYPES
    ]


def rerun_after_delete() -> None:
    st.rerun()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

st.sidebar.markdown(
    """
    <div style="padding:10px 4px 16px">
      <div style="font-size:.72rem;letter-spacing:.18em;color:#a78bfa;font-weight:800">
        INTELLIGENT RESEARCH LAB
      </div>
      <div style="font-size:1.45rem;font-weight:850;margin-top:7px;color:white">
        ◈ Research Assistant
      </div>
      <div style="color:#9ca3af;font-size:.84rem;margin-top:5px">
        Evidence-grounded document intelligence
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

info = vector_store_info()
status_pill("AI research assistant", "online")
st.sidebar.caption(
    f"{count_documents()} documents • {info['vectors']} vectors • {len(get_notes())} notes"
)

page = st.sidebar.radio(
    "Workspace",
    [
        "🏠 Home",
        "📄 Upload",
        "🔎 Semantic Search",
        "💬 Question Answering",
        "🧠 Research Tools",
        "📈 Dataset Analysis",
        "📝 Notes",
        "📊 Dashboard",
        "📤 Export",
    ],
)




# -------------------------------------------------------------------
# Home
# -------------------------------------------------------------------

if page == "🏠 Home":
    hero(
        "Professional AI Research Assistant",
        (
            "Upload academic papers and datasets, retrieve evidence with "
            "Sentence Transformers and FAISS, generate grounded answers with "
            "FLAN-T5, translate with NLLB-200 and analyze structured data with Pandas."
        ),
    )

    info = vector_store_info()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Documents", count_documents())
    metric_columns[1].metric("Vector Embeddings", info["vectors"])
    metric_columns[2].metric("Indexed Chunks", info["chunks"])
    metric_columns[3].metric("Research Notes", len(get_notes()))

    section_header(
        "Research workflow",
        "A complete local retrieval and generation pipeline.",
    )

    workflow_columns = st.columns(6)
    workflow_items = [
        ("Upload", "PDF, DOCX, TXT, PPTX, CSV and XLSX", "01"),
        ("Extract", "Readable document text and structured data", "02"),
        ("Chunk", "Overlapping semantic research passages", "03"),
        ("Embed", "Sentence Transformer vector representations", "04"),
        ("Retrieve", "FAISS similarity search and filtering", "05"),
        ("Generate", "Grounded FLAN-T5 answers and research outputs", "06"),
    ]

    for column, (title, description, icon) in zip(workflow_columns, workflow_items):
        with column:
            info_card(title, description, icon)

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        section_header("Recent documents", "Your active research corpus.")

        documents = get_documents()

        if not documents:
            st.info("Upload your first paper or dataset to begin.")
        else:
            recent = pd.DataFrame(
                documents[:8],
                columns=["ID", "Filename", "Uploaded At"],
            )
            recent["Type"] = recent["Filename"].apply(
                lambda name: Path(name).suffix.upper().replace(".", "")
            )
            st.dataframe(
                recent[["Filename", "Type", "Uploaded At"]],
                use_container_width=True,
                hide_index=True,
            )

    with right:
        section_header("System status", "Current local AI components.")
        info_card("Semantic Retrieval", "all-MiniLM-L6-v2 + FAISS vector index", "◎")
        info_card("Academic Generation", "FLAN-T5 direct model.generate() workflow", "✦")
        info_card("Multilingual Translation", "NLLB-200 with explicit language tokens", "文")


# -------------------------------------------------------------------
# Upload
# -------------------------------------------------------------------

elif page == "📄 Upload":
    hero(
        "Document Workspace",
        "Build a private research corpus from papers, presentations and datasets.",
        "DOCUMENT INGESTION",
    )

    uploaded = st.file_uploader(
        "Drop a research file here",
        type=["pdf", "docx", "txt", "pptx", "csv", "xlsx"],
        help="Maximum upload size is configured in .streamlit/config.toml.",
    )

    process_column, help_column = st.columns([1, 2])

    with process_column:
        process_clicked = st.button(
            "Process document",
            use_container_width=True,
            disabled=uploaded is None,
        )

    with help_column:
        st.caption(
            "PDF/DOCX/TXT/PPTX files are embedded in FAISS. "
            "CSV/XLSX files use structured Pandas search and question answering."
        )

    if uploaded is not None and process_clicked:
        path = UPLOAD_FOLDER / uploaded.name

        if path.exists():
            st.warning("A file with this name already exists.")
        else:
            try:
                progress = st.progress(0, text="Saving document...")
                path.write_bytes(uploaded.getbuffer())
                progress.progress(20, text="Extracting readable content...")

                extension = path.suffix.lower()
                extracted_text = extract_text(str(path))
                progress.progress(45, text="Preparing document intelligence...")

                if extension in TEXT_TYPES:
                    chunks = chunk_text(extracted_text)
                    progress.progress(65, text="Creating semantic embeddings...")
                    embeddings = create_embeddings(chunks)
                    progress.progress(85, text="Updating FAISS vector database...")
                    add_documents(chunks, embeddings, uploaded.name)
                    indexed_message = f"{len(chunks)} semantic chunks indexed."
                else:
                    indexed_message = "Dataset registered for Pandas-based search and analysis."

                save_document(uploaded.name)
                progress.progress(100, text="Processing complete.")
                st.success(f"{uploaded.name} processed successfully. {indexed_message}")

            except Exception as error:
                path.unlink(missing_ok=True)
                st.error(f"Document processing failed: {error}")

    section_header("Document library", "Review, download or remove uploaded files.")

    documents = get_documents()

    if not documents:
        st.info("No documents have been uploaded.")
    else:
        for document_id, filename, uploaded_at in documents:
            path = UPLOAD_FOLDER / filename
            extension = path.suffix.upper().replace(".", "")
            icon = "▧" if path.suffix.lower() in DATASET_TYPES else "▤"

            with st.expander(f"{icon} {filename} · {extension}"):
                st.caption(f"Uploaded: {uploaded_at}")

                action_left, action_right = st.columns(2)

                with action_left:
                    if path.exists():
                        st.download_button(
                            "Download file",
                            data=path.read_bytes(),
                            file_name=filename,
                            key=f"download_document_{document_id}",
                            use_container_width=True,
                        )

                with action_right:
                    confirmed = st.checkbox(
                        "Confirm deletion",
                        key=f"confirm_document_delete_{document_id}",
                    )
                    if st.button(
                        "Delete document",
                        key=f"delete_document_{document_id}",
                        disabled=not confirmed,
                        use_container_width=True,
                    ):
                        delete_document(document_id)
                        path.unlink(missing_ok=True)
                        rerun_after_delete()


# -------------------------------------------------------------------
# Semantic Search
# -------------------------------------------------------------------

elif page == "🔎 Semantic Search":
    hero(
        "Semantic Search",
        "Search selected research papers by meaning or query structured datasets by columns and records.",
        "RETRIEVAL INTELLIGENCE",
    )

    documents = existing_names()

    if not documents:
        st.warning("Upload a document first.")
        st.stop()

    control_left, control_right = st.columns([2, 1])

    with control_left:
        selected_document = st.selectbox(
            "Search in document",
            documents,
            key="semantic_selected_document",
        )
        query = st.text_input(
            "Search query",
            placeholder="Example: What methodology does the study use?",
            key="semantic_query",
        )

    with control_right:
        mode = st.radio(
            "Search method",
            ["Semantic Search", "TF-IDF Search"],
            horizontal=False,
            key="semantic_mode",
        )
        top_k = st.slider(
            "Number of results",
            min_value=3,
            max_value=15,
            value=8,
            key="semantic_top_k",
        )

    suggested_queries = [
        "What is the objective of the study?",
        "Which methodology is used?",
        "What are the main findings?",
        "Which columns are related to health?",
    ]
    st.caption("Suggested queries: " + " • ".join(suggested_queries))

    if st.button("Search selected document", use_container_width=True):
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            extension = Path(selected_document).suffix.lower()

            try:
                if extension in DATASET_TYPES:
                    with st.spinner("Searching structured dataset content..."):
                        dataset_result = search_dataset(
                            filepath=UPLOAD_FOLDER / selected_document,
                            query=query,
                            top_k=top_k,
                        )

                    result_mode = dataset_result.get("mode", "")
                    answer = dataset_result.get("answer", "")
                    results = dataset_result.get("results", [])
                    available_columns = dataset_result.get("available_columns", [])

                    if result_mode in {"structured_answer", "research_field_values"}:
                        st.success(answer)
                    elif result_mode in {"research_field_missing", "research_field_empty"}:
                        st.warning(answer)
                    elif result_mode == "missing_values":
                        st.info(answer)
                        if results:
                            st.dataframe(
                                pd.DataFrame(results),
                                use_container_width=True,
                                hide_index=True,
                            )
                    elif result_mode == "columns":
                        if results:
                            st.success(answer)
                            st.dataframe(
                                pd.DataFrame(results),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.warning(answer)
                    elif result_mode == "rows":
                        if not results:
                            st.warning(answer)
                        else:
                            st.success(answer)
                            for item in results:
                                with st.expander(
                                    f"Row {item.get('row_number', item.get('rank'))} "
                                    f"— {item.get('score_percentage', 0)}%"
                                ):
                                    st.write(item.get("chunk", ""))

                    if result_mode in {"research_field_missing", "columns"} and available_columns:
                        st.caption("Available columns: " + ", ".join(map(str, available_columns)))

                else:
                    if mode == "Semantic Search":
                        with st.spinner("Creating query embedding and searching FAISS..."):
                            raw_results = search_documents(
                                create_query_embedding(query),
                                top_k=max(top_k * 5, 30),
                            )

                        results = [
                            result
                            for result in raw_results
                            if result.get("filename") == selected_document
                            and float(result.get("score", 0.0)) >= 0.15
                        ][:top_k]
                    else:
                        text = read_document(selected_document)
                        chunks = chunk_text(text)
                        metadata = [
                            {
                                "filename": selected_document,
                                "chunk_id": index,
                            }
                            for index, _ in enumerate(chunks, start=1)
                        ]
                        results = tfidf_search(
                            query=query,
                            chunks=chunks,
                            metadata=metadata,
                            top_k=top_k,
                        )
                        results = [
                            result for result in results
                            if float(result.get("score", 0.0)) >= 0.05
                        ]

                    if not results:
                        st.warning(
                            "No sufficiently relevant passage was found in the selected document."
                        )
                    else:
                        for result in results:
                            score = result.get(
                                "score_percentage",
                                round(float(result.get("score", 0.0)) * 100, 2),
                            )
                            with st.expander(
                                f"Rank {result.get('rank', '—')} · "
                                f"{selected_document} · {score}%"
                            ):
                                st.caption(
                                    f"Chunk ID: {result.get('chunk_id', 'Unknown')}"
                                )
                                st.progress(min(max(float(score) / 100, 0.0), 1.0))
                                st.write(result.get("chunk", ""))

            except Exception as error:
                st.error(f"Search failed: {error}")


# -------------------------------------------------------------------
# Question Answering
# -------------------------------------------------------------------

elif page == "💬 Question Answering":
    hero(
        "Question Answering",
        "Research papers use Sentence Transformers, FAISS and FLAN-T5. "
        "CSV/XLSX datasets use deterministic Pandas calculations.",
        "GROUNDED ANSWERS",
    )

    documents = existing_names()

    if not documents:
        st.warning("Upload and process at least one document first.")
        st.stop()

    selected_document = st.selectbox(
        "Select document",
        documents,
        key="qa_selected_document",
    )
    extension = Path(selected_document).suffix.lower()

    if extension in DATASET_TYPES:
        st.info("Dataset mode: exact Pandas-based question answering.")
        suggestions = [
            "How many rows are in the dataset?",
            "What are the column names?",
            "Are there missing values?",
            "What is the average age?",
        ]
    else:
        st.info("Research-document mode: FAISS retrieval followed by grounded FLAN-T5 generation.")
        suggestions = [
            "What is the objective of the study?",
            "What methodology is used?",
            "What are the main findings?",
            "What limitations are discussed?",
        ]

    st.caption("Suggested questions: " + " • ".join(suggestions))

    question = st.text_input(
        "Ask a question",
        placeholder="Ask about the selected document...",
        key="qa_question_input",
    )

    answer_clicked = st.button(
        "Generate grounded answer",
        use_container_width=True,
        key="qa_generate_answer",
    )

    if answer_clicked:
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                filepath = UPLOAD_FOLDER / selected_document

                if extension in DATASET_TYPES:
                    with st.spinner("Analyzing dataset structure and values..."):
                        result = answer_dataset_question(
                            filepath=filepath,
                            question=question,
                        )

                    if result["status"] == "success":
                        st.success(result["answer"])
                    elif result["status"] == "unsupported":
                        st.info(result["answer"])
                    else:
                        st.warning(result["answer"])

                    if result.get("details"):
                        with st.expander("Calculation details"):
                            st.json(result["details"])

                else:
                    with st.spinner("Retrieving evidence and generating a grounded answer..."):
                        result = ask_rag(
                            question=question,
                            selected_document=selected_document,
                            top_k=5,
                            minimum_score=0.25,
                        )

                    st.session_state.chat_history.append({
                        "document": selected_document,
                        "question": question,
                        "answer": result["answer"],
                        "confidence": result["confidence"],
                        "sources": result.get("sources", []),
                    })

                    result_panel(
                        "Document answer",
                        result["answer"],
                        f"Retrieval status: {result.get('status', 'generated')}",
                    )

                    metric_left, metric_right = st.columns(2)
                    metric_left.metric(
                        "Retrieval Confidence",
                        f"{result['confidence']:.2f}%",
                    )
                    metric_right.metric(
                        "Confidence Label",
                        result.get("confidence_label", "Unknown"),
                    )

                    if result.get("sources"):
                        st.write("#### Sources")
                        st.write(" • ".join(result["sources"]))

                    if result.get("evidence"):
                        st.write("#### Retrieved Evidence")

                        for item in result["evidence"]:
                            score = item.get(
                                "score_percentage",
                                round(float(item.get("score", 0.0)) * 100, 2),
                            )
                            with st.expander(
                                f"{item.get('filename', 'Unknown')} · "
                                f"Chunk {item.get('chunk_id', 'Unknown')} · {score}%"
                            ):
                                st.write(item.get("chunk", ""))

            except Exception as error:
                st.error(f"Question answering failed: {error}")

    if st.session_state.chat_history:
        st.divider()
        section_header("Recent questions", "Answers generated during this session.")

        for item in reversed(st.session_state.chat_history[-5:]):
            with st.expander(item["question"]):
                st.write(item["answer"])
                st.caption(
                    f"{item['document']} · {item['confidence']:.2f}% confidence"
                )

        if st.button("Clear question history"):
            st.session_state.chat_history = []
            st.rerun()


# -------------------------------------------------------------------
# Research Tools
# -------------------------------------------------------------------

elif page == "🧠 Research Tools":
    hero(
        "Research Intelligence Tools",
        "Create summaries, abstracts, keywords, research questions, translations, "
        "citations, literature reviews and paper comparisons.",
        "ACADEMIC GENERATION",
    )

    documents = text_documents()

    if not documents:
        st.warning("Upload a PDF, DOCX, TXT or PPTX research document first.")
        st.stop()

    selected_document = st.selectbox(
        "Primary research document",
        documents,
        key="research_selected_document",
    )
    text = read_document(selected_document)

    tabs = st.tabs([
        "Summary",
        "Abstract",
        "Keywords",
        "Research Questions",
        "Translation",
        "Citation",
        "Literature Review",
        "Paper Comparison",
        "Similarity",
        "Voice",
    ])

    with tabs[0]:
        section_header("Academic Summary", "Map-reduce summarization for longer papers.")
        summary_type = st.selectbox(
            "Summary type",
            ["Short", "Detailed", "Bullet Points"],
            key="summary_type",
        )

        if st.button("Generate summary", key="generate_summary", use_container_width=True):
            with st.spinner("Summarizing document sections and combining the result..."):
                try:
                    st.session_state.summary = generate_summary(
                        text=text,
                        summary_type=summary_type,
                    )
                except Exception as error:
                    st.error(f"Summary generation failed: {error}")

        if st.session_state.summary:
            result_panel("Generated Summary", st.session_state.summary)
            st.download_button(
                "Download summary",
                st.session_state.summary,
                file_name=f"{Path(selected_document).stem}_summary.txt",
            )

    with tabs[1]:
        section_header("Academic Abstract", "Generate a structured research abstract.")

        if st.button("Generate abstract", key="generate_abstract", use_container_width=True):
            with st.spinner("Generating academic abstract..."):
                try:
                    st.session_state.abstract = generate_abstract(text)
                except Exception as error:
                    st.error(f"Abstract generation failed: {error}")

        if st.session_state.abstract:
            result_panel("Generated Abstract", st.session_state.abstract)

    with tabs[2]:
        section_header("Keyword Extraction", "TF-IDF unigram, bigram and trigram key phrases.")
        keyword_count = st.slider(
            "Keyword count",
            min_value=5,
            max_value=30,
            value=15,
            key="keyword_count",
        )

        if st.button("Extract keywords", key="extract_keywords", use_container_width=True):
            try:
                st.session_state.keywords = extract_keywords(
                    text=text,
                    keyword_count=keyword_count,
                )
            except Exception as error:
                st.error(f"Keyword extraction failed: {error}")

        if st.session_state.keywords:
            st.write("#### Extracted Key Phrases")
            st.write(" • ".join(st.session_state.keywords))

    with tabs[3]:
        section_header(
            "Research Question Generation",
            "The slider controls the exact number of questions returned.",
        )
        question_count = st.slider(
            "Question count",
            min_value=3,
            max_value=20,
            value=10,
            key="research_question_count",
        )

        if st.button(
            "Generate research questions",
            key="generate_research_questions",
            use_container_width=True,
        ):
            with st.spinner("Generating distinct academic questions..."):
                try:
                    st.session_state.questions = generate_research_questions(
                        text=text,
                        question_count=question_count,
                    )
                except Exception as error:
                    st.error(f"Research-question generation failed: {error}")

        if st.session_state.questions:
            questions = [
                line.strip()
                for line in st.session_state.questions.splitlines()
                if line.strip()
            ]
            for question in questions:
                st.markdown(
                    f'<div class="glass-card" style="margin-bottom:10px">{question}</div>',
                    unsafe_allow_html=True,
                )

    with tabs[4]:
        section_header("Multilingual Translation", "NLLB-200 local translation.")

        languages = supported_languages()
        source_language = st.selectbox(
            "Source language",
            languages,
            index=languages.index("English") if "English" in languages else 0,
            key="translation_source",
        )
        target_options = [
            language for language in languages
            if language != source_language
        ]
        target_language = st.selectbox(
            "Target language",
            target_options,
            key="translation_target",
        )
        translation_scope = st.radio(
            "Translate",
            ["Full Document", "Summary", "Abstract", "Custom Text"],
            horizontal=True,
            key="translation_scope",
        )

        if translation_scope == "Full Document":
            translation_input = text
        elif translation_scope == "Summary":
            translation_input = st.session_state.summary
        elif translation_scope == "Abstract":
            translation_input = st.session_state.abstract
        else:
            translation_input = st.text_area(
                "Custom text",
                height=180,
                key="translation_custom_text",
            )

        st.caption("The first translation downloads the NLLB-200 model and can take time.")

        if st.button("Translate text", key="translate_text", use_container_width=True):
            if not str(translation_input).strip():
                st.warning("No text is available for translation.")
            else:
                with st.spinner(f"Translating into {target_language}..."):
                    try:
                        st.session_state.translation = translate_text(
                            text=translation_input,
                            source_language=source_language,
                            target_language=target_language,
                        )
                    except Exception as error:
                        st.error(f"Translation failed: {error}")

        if st.session_state.translation:
            result_panel(
                f"{target_language} Translation",
                st.session_state.translation,
            )
            st.download_button(
                "Download translation",
                st.session_state.translation,
                file_name=f"translation_{target_language}.txt",
            )

    with tabs[5]:
        section_header("Citation Generator", "Basic citation formatting from file metadata.")
        style = st.selectbox(
            "Citation style",
            ["APA", "MLA", "IEEE", "Harvard", "BibTeX"],
            key="citation_style",
        )

        if st.button("Generate citation", key="generate_citation", use_container_width=True):
            try:
                st.session_state.citation = generate_citation(
                    selected_document,
                    style,
                )
            except Exception as error:
                st.error(f"Citation generation failed: {error}")

        if st.session_state.citation:
            st.code(st.session_state.citation)

    with tabs[6]:
        section_header("Literature Review", "Synthesize multiple selected documents.")
        selected_documents = st.multiselect(
            "Select research documents",
            documents,
            key="literature_documents",
        )
        topic = st.text_input(
            "Literature review topic",
            key="literature_topic",
        )

        if st.button(
            "Generate literature review",
            key="generate_literature_review",
            use_container_width=True,
        ):
            if not selected_documents:
                st.warning("Select at least one document.")
            else:
                with st.spinner("Synthesizing selected research documents..."):
                    try:
                        payload = [
                            {
                                "filename": name,
                                "text": read_document(name),
                            }
                            for name in selected_documents
                        ]
                        st.session_state.literature_review = generate_literature_review(
                            payload,
                            topic,
                        )
                    except Exception as error:
                        st.error(f"Literature review failed: {error}")

        if st.session_state.literature_review:
            result_panel(
                "Literature Review",
                st.session_state.literature_review,
            )

    with tabs[7]:
        section_header("Paper Comparison", "Compare at least two research documents.")
        selected_documents = st.multiselect(
            "Select papers",
            documents,
            key="comparison_documents",
        )

        if st.button(
            "Compare selected papers",
            key="compare_papers",
            use_container_width=True,
        ):
            if len(selected_documents) < 2:
                st.warning("Select at least two research documents.")
            else:
                with st.spinner("Comparing selected papers..."):
                    try:
                        payload = [
                            {
                                "filename": name,
                                "text": read_document(name),
                            }
                            for name in selected_documents
                        ]
                        st.session_state.paper_comparison = compare_papers(payload)
                    except Exception as error:
                        st.error(f"Paper comparison failed: {error}")

        if st.session_state.paper_comparison:
            result_panel(
                "Paper Comparison",
                st.session_state.paper_comparison,
            )

    with tabs[8]:
        section_header("Document Similarity", "TF-IDF cosine similarity between two papers.")

        first_document = st.selectbox(
            "First document",
            documents,
            key="similarity_first",
        )
        second_options = [
            name for name in documents
            if name != first_document
        ]

        if not second_options:
            st.info("Upload a second research document to calculate similarity.")
        else:
            second_document = st.selectbox(
                "Second document",
                second_options,
                key="similarity_second",
            )

            if st.button(
                "Calculate similarity",
                key="calculate_similarity",
                use_container_width=True,
            ):
                try:
                    similarity = compare_similarity(
                        read_document(first_document),
                        read_document(second_document),
                    )
                    st.metric("Document Similarity", f"{similarity:.2f}%")
                except Exception as error:
                    st.error(f"Similarity calculation failed: {error}")

    with tabs[9]:
        section_header("Voice Output", "Convert generated research content into audio.")
        voice_source = st.selectbox(
            "Content source",
            ["Summary", "Abstract", "Translation", "Custom Text"],
            key="voice_source",
        )
        custom_voice = (
            st.text_area("Custom voice text", key="voice_custom")
            if voice_source == "Custom Text"
            else ""
        )
        voice_text = {
            "Summary": st.session_state.summary,
            "Abstract": st.session_state.abstract,
            "Translation": st.session_state.translation,
            "Custom Text": custom_voice,
        }[voice_source]

        if st.button("Generate audio", key="generate_audio", use_container_width=True):
            if not str(voice_text).strip():
                st.warning("No text is available for voice generation.")
            else:
                with st.spinner("Generating audio..."):
                    try:
                        audio_path = text_to_speech(voice_text)
                        audio_bytes = Path(audio_path).read_bytes()
                        st.audio(audio_bytes, format="audio/mp3")
                        st.download_button(
                            "Download audio",
                            audio_bytes,
                            file_name="research_audio.mp3",
                        )
                    except Exception as error:
                        st.error(f"Voice generation failed: {error}")


# -------------------------------------------------------------------
# Dataset Analysis
# -------------------------------------------------------------------

elif page == "📈 Dataset Analysis":
    hero(
        "Dataset Analytics",
        "Explore CSV and Excel datasets with descriptive statistics, data-quality "
        "checks, group comparisons and interactive Plotly visualizations.",
        "STRUCTURED DATA INTELLIGENCE",
    )

    datasets = dataset_documents()

    if not datasets:
        st.warning("Upload a CSV or XLSX dataset first.")
        st.stop()

    selected_dataset = st.selectbox(
        "Select dataset",
        datasets,
        key="analysis_dataset",
    )

    try:
        dataframe = load_dataset(UPLOAD_FOLDER / selected_dataset)
        overview = get_dataset_overview(dataframe)

        metrics = st.columns(4)
        metrics[0].metric("Rows", overview["rows"])
        metrics[1].metric("Columns", overview["columns"])
        metrics[2].metric("Missing Values", overview["missing_values"])
        metrics[3].metric("Duplicate Rows", overview["duplicate_rows"])

        tabs = st.tabs([
            "Preview",
            "Statistics",
            "Data Quality",
            "Distributions",
            "Group Comparison",
            "Correlation",
        ])

        with tabs[0]:
            st.dataframe(
                dataframe.head(200),
                use_container_width=True,
                hide_index=True,
            )

        with tabs[1]:
            st.dataframe(
                dataframe.describe(include="all").transpose(),
                use_container_width=True,
            )

        with tabs[2]:
            missing = dataframe.isna().sum().reset_index()
            missing.columns = ["Column", "Missing Values"]
            missing["Missing Percentage"] = (
                missing["Missing Values"] / max(len(dataframe), 1) * 100
            ).round(2)
            st.dataframe(missing, use_container_width=True, hide_index=True)

        with tabs[3]:
            numeric_columns = overview["numeric_columns"]
            categorical_columns = overview["categorical_columns"]

            distribution_left, distribution_right = st.columns(2)

            with distribution_left:
                if numeric_columns:
                    selected_numeric = st.selectbox(
                        "Numeric distribution",
                        numeric_columns,
                        key="distribution_numeric",
                    )
                    figure = px.histogram(
                        dataframe,
                        x=selected_numeric,
                        marginal="box",
                        title=f"Distribution of {selected_numeric}",
                    )
                    figure.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                    )
                    st.plotly_chart(figure, use_container_width=True)

            with distribution_right:
                if categorical_columns:
                    selected_category = st.selectbox(
                        "Category distribution",
                        categorical_columns,
                        key="distribution_category",
                    )
                    counts = (
                        dataframe[selected_category]
                        .astype(str)
                        .value_counts()
                        .head(20)
                        .reset_index()
                    )
                    counts.columns = [selected_category, "Count"]
                    figure = px.bar(
                        counts,
                        x=selected_category,
                        y="Count",
                        text="Count",
                        title=f"Top values in {selected_category}",
                    )
                    figure.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                    )
                    st.plotly_chart(figure, use_container_width=True)

        with tabs[4]:
            numeric_columns = overview["numeric_columns"]
            categorical_columns = overview["categorical_columns"]

            if numeric_columns and categorical_columns:
                comparison_left, comparison_right = st.columns(2)
                with comparison_left:
                    value_column = st.selectbox(
                        "Numeric column",
                        numeric_columns,
                        key="group_value_column",
                    )
                with comparison_right:
                    group_column = st.selectbox(
                        "Group column",
                        categorical_columns,
                        key="group_group_column",
                    )

                comparison = compare_numeric_by_group(
                    dataframe,
                    value_column,
                    group_column,
                )
                st.dataframe(comparison, use_container_width=True, hide_index=True)

                chart_left, chart_right = st.columns(2)

                with chart_left:
                    figure = px.bar(
                        comparison,
                        x=group_column,
                        y="Mean",
                        text="Mean",
                        title=f"Mean {value_column} by {group_column}",
                    )
                    figure.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                    )
                    st.plotly_chart(figure, use_container_width=True)

                with chart_right:
                    figure = px.box(
                        dataframe,
                        x=group_column,
                        y=value_column,
                        points="outliers",
                        title=f"{value_column} distribution by {group_column}",
                    )
                    figure.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                    )
                    st.plotly_chart(figure, use_container_width=True)
            else:
                st.info("A numeric and categorical column are required.")

        with tabs[5]:
            numeric = dataframe.select_dtypes(include="number")

            if numeric.empty:
                st.info("No numeric columns are available.")
            else:
                figure = px.imshow(
                    numeric.corr(),
                    text_auto=".2f",
                    aspect="auto",
                    title="Correlation Matrix",
                )
                figure.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                )
                st.plotly_chart(figure, use_container_width=True)

    except Exception as error:
        st.error(f"Dataset analysis failed: {error}")


# -------------------------------------------------------------------
# Notes
# -------------------------------------------------------------------

elif page == "📝 Notes":
    hero(
        "Research Notes",
        "Capture findings, methods, conclusions and ideas linked to your documents.",
        "KNOWLEDGE MANAGEMENT",
    )

    form_left, form_right = st.columns([1, 1])

    with form_left:
        note_title = st.text_input("Note title", key="note_title")
        note_category = st.selectbox(
            "Category",
            ["General", "Summary", "Findings", "Methodology", "Results", "Conclusion"],
            key="note_category",
        )

    with form_right:
        linked_document = st.selectbox(
            "Linked document",
            ["None", *existing_names()],
            key="note_document",
        )
        note_tags = st.text_input(
            "Tags",
            placeholder="rag, healthcare, methods",
            key="note_tags",
        )

    note_content = st.text_area(
        "Note content",
        height=180,
        key="note_content",
    )

    if st.button("Save research note", use_container_width=True):
        if not note_title.strip() or not note_content.strip():
            st.warning("Title and content are required.")
        else:
            create_note(
                note_title,
                note_content,
                note_category,
                "" if linked_document == "None" else linked_document,
                note_tags,
            )
            st.success("Note saved.")
            st.rerun()

    search_notes = st.text_input(
        "Search saved notes",
        key="search_notes",
    )

    notes = get_notes(search_notes)

    if not notes:
        st.info("No matching notes were found.")
    else:
        for note in notes:
            with st.expander(f"✎ {note['title']} · {note['category']}"):
                st.write(note["content"])
                st.caption(
                    f"{note['linked_document'] or 'No document'} · "
                    f"{note['tags'] or 'No tags'} · {note['created_at']}"
                )
                confirmed = st.checkbox(
                    "Confirm deletion",
                    key=f"confirm_note_delete_{note['id']}",
                )
                if st.button(
                    "Delete note",
                    key=f"delete_note_{note['id']}",
                    disabled=not confirmed,
                ):
                    delete_note(note["id"])
                    st.rerun()


# -------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------

elif page == "📊 Dashboard":
    hero(
        "Research Dashboard",
        "Monitor documents, vector-store activity, file types and research notes.",
        "WORKSPACE ANALYTICS",
    )

    documents = get_documents()
    notes = get_notes()
    info = vector_store_info()

    metrics = st.columns(4)
    metrics[0].metric("Documents", len(documents))
    metrics[1].metric("Vectors", info["vectors"])
    metrics[2].metric("Chunks", info["chunks"])
    metrics[3].metric("Notes", len(notes))

    if documents:
        document_frame = pd.DataFrame(
            documents,
            columns=["ID", "Filename", "Uploaded At"],
        )
        document_frame["Type"] = document_frame["Filename"].apply(
            lambda name: Path(name).suffix.upper().replace(".", "")
        )
        document_frame["Uploaded At"] = pd.to_datetime(
            document_frame["Uploaded At"],
            errors="coerce",
        )

        type_counts = document_frame["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]

        chart_left, chart_right = st.columns(2)

        with chart_left:
            figure = px.pie(
                type_counts,
                names="Type",
                values="Count",
                hole=.55,
                title="File Type Distribution",
            )
            figure.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(figure, use_container_width=True)

        with chart_right:
            timeline = (
                document_frame.dropna(subset=["Uploaded At"])
                .assign(Date=lambda frame: frame["Uploaded At"].dt.date)
                .groupby("Date")
                .size()
                .reset_index(name="Uploads")
            )
            figure = px.line(
                timeline,
                x="Date",
                y="Uploads",
                markers=True,
                title="Upload Timeline",
            )
            figure.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="white",
            )
            st.plotly_chart(figure, use_container_width=True)

        st.dataframe(
            document_frame[["Filename", "Type", "Uploaded At"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Upload documents to populate dashboard analytics.")


# -------------------------------------------------------------------
# Export
# -------------------------------------------------------------------

elif page == "📤 Export":
    hero(
        "Research Report Export",
        "Combine generated research outputs into TXT, DOCX or PDF reports.",
        "REPORT PUBLISHING",
    )

    report_title = st.text_input(
        "Report title",
        value="AI Research Report",
        key="report_title",
    )
    filename = st.text_input(
        "Filename",
        value="research_report",
        key="report_filename",
    )

    sections = {
        "Summary": st.session_state.summary,
        "Abstract": st.session_state.abstract,
        "Keywords": ", ".join(st.session_state.keywords),
        "Research Questions": st.session_state.questions,
        "Translation": st.session_state.translation,
        "Citation": st.session_state.citation,
        "Literature Review": st.session_state.literature_review,
        "Paper Comparison": st.session_state.paper_comparison,
        "Main Findings": st.session_state.main_findings,
        "Limitations": st.session_state.limitations,
        "Conclusion": st.session_state.conclusion,
    }

    available_sections = [
        name for name, value in sections.items()
        if str(value).strip()
    ]

    selected_sections = st.multiselect(
        "Select report sections",
        list(sections.keys()),
        default=available_sections,
        key="report_sections",
    )
    report_format = st.radio(
        "Export format",
        ["TXT", "DOCX", "PDF"],
        horizontal=True,
        key="report_format",
    )

    preview_content = {
        name: sections[name]
        for name in selected_sections
        if str(sections[name]).strip()
    }

    if preview_content:
        with st.expander("Report preview"):
            for section_name, section_content in preview_content.items():
                st.write(f"### {section_name}")
                st.write(section_content)
    else:
        st.info("Generate research-tool outputs and select at least one section.")

    if st.button("Generate report", use_container_width=True):
        if not report_title.strip() or not filename.strip():
            st.warning("Report title and filename are required.")
        elif not preview_content:
            st.warning("Select at least one non-empty section.")
        else:
            try:
                with st.spinner("Generating professional research report..."):
                    exporter = {
                        "TXT": export_txt,
                        "DOCX": export_docx,
                        "PDF": export_pdf,
                    }[report_format]
                    report_path = exporter(
                        report_title,
                        preview_content,
                        filename,
                    )
                    st.session_state.last_report_path = str(report_path)

                st.success(f"{report_path.name} generated successfully.")
                st.download_button(
                    "Download generated report",
                    data=report_path.read_bytes(),
                    file_name=report_path.name,
                    use_container_width=True,
                )

            except Exception as error:
                st.error(f"Report generation failed: {error}")
