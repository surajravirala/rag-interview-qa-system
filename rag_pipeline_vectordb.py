
import os
GOOGLE_API_KEY = "#"  # CHANGE THIS!

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from langchain_community.document_loaders import DirectoryLoader, TextLoader


import os
import zipfile

DATA_PATH = "./data/Interview_questions"  # Update this path

# Define the path to the zip file and the target directory
ZIP_FILE_PATH = "/content/Interview_questions.zip"
TARGET_DIR = "./data"

# Create the target directory if it doesn't exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Unzip the file if it exists
if os.path.exists(ZIP_FILE_PATH):
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    print(f"✅ Unzipped '{ZIP_FILE_PATH}' to '{TARGET_DIR}'")
else:
    print(f"⚠️ Zip file not found: {ZIP_FILE_PATH}. Please ensure it's uploaded or check the path.")

print(f"Loading documents from: {DATA_PATH}")

# Load Data
loader = DirectoryLoader(
    DATA_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader
)

documents = loader.load()

print(f" Loaded {len(documents)} documents")

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

print(f" Created {len(texts)} text chunks")

# View first chunk (optional)
print("\nFirst chunk preview:")
print(texts[0].page_content[:200] + "...")
len(texts)
print("___________")
texts[0]



embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'},  # Change to 'cpu' if no GPU
    encode_kwargs={'normalize_embeddings': True}
)



vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory="./chroma_db"
)

print(f"✅ Vector database created with {len(texts)} vectors")

retriever=vectordb.as_retriever()
docs=retriever.invoke("Tell me about Numphy")
len(docs)

docs

retriever=vectordb.as_retriever(search_kwargs={"k":2})

retriever

retriever.search_type



llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    convert_system_message_to_human=True
)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
qa_chain("what is python")

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

prompt = "What is Python"
llm_response = qa_chain(prompt)
process_llm_response(llm_response)




print("\nAvailable Gemini models (supporting generateContent):")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods and "gemini" in m.name:
        print(m.name)
