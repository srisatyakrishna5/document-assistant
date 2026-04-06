# Document Assistant

Document Assistant is a Streamlit-based Retrieval Augmented Generation application for asking questions about PDF documents indexed in Azure AI Search.

The current application supports:

- PDF upload and indexing through Azure Document Intelligence
- Hybrid retrieval over indexed chunks with Azure AI Search
- Grounded answers with GPT-4.1 and page-level citations
- Whole-document summaries from the Summary tab

The current application does not include voice input, text-to-speech, or a visible language selector.

---

## Overview

The end-to-end flow is:

1. Upload a PDF.
2. Extract page text with Azure Document Intelligence.
3. Split the extracted text into overlapping chunks.
4. Generate embeddings with Azure OpenAI.
5. Store chunks and vectors in Azure AI Search.
6. Use chat or summary generation against the indexed content.

---

## Architecture

### Document ingestion

```text
PDF upload
  -> Azure Document Intelligence
  -> chunking
  -> Azure OpenAI embeddings
  -> Azure AI Search index
```

### Question answering

```text
User question
  -> hybrid search (BM25 + vector)
  -> top matching chunks
  -> GPT-4.1 answer generation
  -> cited response in the UI
```

### Summary generation

```text
Selected document
  -> fetch all indexed chunks
  -> GPT-4.1 summary generation
  -> rendered summary in the UI
```

---

## Features

### PDF upload and indexing

- Upload a PDF from the sidebar.
- Preview extracted text for the first few pages.
- Create the Azure AI Search index automatically if needed.
- Upload chunk text and embeddings to the configured index.

### Chat with citations

- Ask questions in the Chat tab.
- Retrieve the most relevant indexed chunks with hybrid search.
- Generate grounded answers with GPT-4.1.
- Inspect supporting passages in the Sources & Citations expander.

### Document summary

- Select an indexed document from the Summary tab.
- Generate a structured summary over the full indexed document.
- Clear the cached summary without re-indexing the document.

---

## Repository structure

```text
document-assistant/
├── app.py
├── config.py
├── LAB_MANUAL.md
├── README.md
├── requirements.txt
├── docs/
│   ├── .nojekyll
│   └── index.html
└── services/
    ├── __init__.py
    ├── chunker.py
    ├── document_intelligence.py
    ├── embeddings.py
    ├── llm.py
    ├── search.py
    ├── search_index.py
    └── translation.py
```

### Key modules

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit UI, session state, upload flow, chat flow, and summary flow |
| `config.py` | Environment variable loading and shared constants |
| `services/document_intelligence.py` | PDF analysis with Azure Document Intelligence |
| `services/chunker.py` | Chunk generation from extracted page text |
| `services/embeddings.py` | Azure OpenAI embedding generation |
| `services/search_index.py` | Search index creation and chunk upload |
| `services/search.py` | Hybrid search and document retrieval helpers |
| `services/llm.py` | GPT-4.1 answer and summary generation |
| `services/translation.py` | Translation helper used by backend code |

---

## Azure resources required

The current application requires:

| Service | Purpose |
|---|---|
| Azure Document Intelligence | Extract text from uploaded PDFs |
| Azure AI Search | Store and retrieve indexed chunks |
| Microsoft Foundry or Azure OpenAI | Host GPT-4.1 and text-embedding-3-large deployments |

Expected model deployments:

- `gpt-4.1`
- `text-embedding-3-large`

---

## Prerequisites

- Python 3.10 or later
- An Azure subscription
- A provisioned Document Intelligence resource
- A provisioned Azure AI Search resource
- A Microsoft Foundry project or Azure OpenAI resource with:
  - a `gpt-4.1` deployment
  - a `text-embedding-3-large` deployment

---

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd document-assistant
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.env`

Create a `.env` file in the project root:

```dotenv
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://YOUR-RESOURCE-NAME.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=PASTE_YOUR_KEY_HERE

AZURE_SEARCH_ENDPOINT=https://YOUR-SEARCH-NAME.search.windows.net
AZURE_SEARCH_KEY=PASTE_YOUR_ADMIN_KEY_HERE
AZURE_SEARCH_INDEX_NAME=rag-documents

AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_KEY=PASTE_YOUR_API_KEY_HERE
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
AZURE_OPENAI_API_VERSION=2025-03-01-preview
```

### 5. Run the app

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.

---

## Environment variables

### Required

| Variable | Description |
|---|---|
| `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` | Document Intelligence endpoint |
| `AZURE_DOCUMENT_INTELLIGENCE_KEY` | Document Intelligence key |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search endpoint |
| `AZURE_SEARCH_KEY` | Azure AI Search admin key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI or Foundry OpenAI endpoint |
| `AZURE_OPENAI_KEY` | Azure OpenAI or Foundry API key |

### Optional or defaulted

| Variable | Default | Description |
|---|---|---|
| `AZURE_SEARCH_INDEX_NAME` | `rag-documents` | Search index name |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-4.1` | Chat model deployment name |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | `text-embedding-3-large` | Embedding deployment name |
| `AZURE_OPENAI_API_VERSION` | `2025-03-01-preview` | API version used by the app |

---

## Using the application

### Upload and index a document

1. Open the sidebar.
2. Choose a PDF file.
3. Click `Analyze & Index`.
4. Wait for indexing to complete.

### Ask questions

1. Open the Chat tab.
2. Enter a question in the chat input.
3. Review the generated answer.
4. Expand Sources & Citations to inspect the retrieved chunks.

### Generate a summary

1. Open the Summary tab.
2. Select a document.
3. Click `Generate Summary`.

---

## How RAG works here

During indexing, the application extracts page text, breaks it into chunks, generates embeddings, and stores both chunk text and vectors in Azure AI Search.

During chat, the application runs hybrid retrieval to find the most relevant chunks for the user question, then sends those chunks to GPT-4.1 as grounding context. The answer returned to the UI is paired with the retrieved supporting passages.

During summary generation, the application fetches all chunks for one document and asks GPT-4.1 to synthesize a document-level summary.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Missing Azure configuration on startup | One or more required `.env` values are missing | Fill in the required environment variables and restart the app |
| Document Intelligence error during upload | Invalid endpoint, invalid key, or problematic PDF | Verify credentials and try another PDF |
| Search index error during upload | Search service misconfiguration or insufficient permissions | Use an admin key and verify the search service is available |
| Indexing error while uploading chunks | Embedding or search upload failed | Verify OpenAI deployment names and search credentials |
| No documents found in Summary | Nothing has been indexed yet, or the index is empty | Upload and index a document first |
| Answers are slow | Model latency during retrieval and generation | This is expected for larger prompts and documents |

---

## Additional documentation

- `LAB_MANUAL.md` contains the non-technical lab walkthrough.
- `docs/index.html` is the published HTML version of the lab manual.
