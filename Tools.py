from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain_community.cache import InMemoryCache
from data_loader import PDFTextProcessor
import time
import os
from glob import glob
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# Load the OpenAI API key
load_dotenv("secret_keys.env")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
openai_api_key = os.getenv("OPENAI_API_KEY")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, streaming=True,api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# Connect to a Pinecone index
index_name = "rc-index"

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
index = pc.Index(index_name)


RAG_DATA="pmres/*.pdf"
pdf_files = glob(RAG_DATA)
cvs = []
for pdf_file in pdf_files:
    processor = PDFTextProcessor(pdf_file)
    cv = processor.process()
    cvs.append(cv)


# Initialize cache
cache = InMemoryCache()

documents = [
    Document(
        page_content=cv,
        metadata={"source": "PM Accelerator"}
    )
    for cv in cvs
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_texts = text_splitter.split_documents(documents)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
vector_store.add_documents(documents=split_texts)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

def create_cache_key(input_text):
    """
    Creates a consistent cache key from input text.

    Args:
        input_text (str): The input text to use as a cache key.

    Returns:
        str: The formatted cache key.
    """
    return f"RAG_QUERY: {input_text}"

@tool
def work_experience_to_pm_focused_entry(input_text):
    """
    Converts an achievement/responsibility into an enhanced, PM-focused entry using retrieved context.
    Implements caching to improve performance for repeated queries.

    Args:
        input_text (str): The original achievement/responsibility text to be enhanced.

    Returns:
        str: The enhanced, PM-focused achievement/responsibility entry.
    """
    # Generate cache prompt
    cache_prompt = create_cache_key(input_text)
    llm_string = str(llm)

    # Check cache first
    cached_result = cache.lookup(cache_prompt, llm_string)
    if cached_result is not None:
        return cached_result

    # Define system prompt template
    system_prompt =(
    "You are an assistant specialized in converting achievements and responsibilities into enhanced, PM-focused entries "
    "that include quantifiable measures. Utilize the provided retrieved context to refine the original achievement or responsibility. "
    "Ensure the final output aligns stylistically with the retrieved context, while incorporating measurable impacts such as "
    "'enhanced the system by X%', 'reduced processing time by Y hours/days', or 'achieved Z% improvement in user engagement.' "
    "Always maintain accuracy and do not introduce non-factual information or unrealistic metrics."
    "\n\n"
    "Provide concise sentence."
    "\n\n"
    "{context}"
)


    # Define chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Create the QA chain using the provided language model and prompt template
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the RAG chain by combining retriever and QA chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Generate new response
    result = rag_chain.invoke({"input": input_text})

    # Cache the new response
    cache.update(cache_prompt, llm_string, result['answer'])

    return result['answer']



@tool
def rewrite_to_PM_focused(input_text):
    """
    Convert any part of a resume into a PM focused part\
    and add PM relevant content

    Args:
          input_text (str): The original part text to be enhanced.

      Returns:
          str: The enhanced, PM-focused content.

    """
    # Define system prompt template
    system_prompt = (
        "You are an assistant specialized in converting resume sections (Professional Summary, Skills, etc.) into enhanced, PM-focused entries"
        "When enhancing sections, focus on PM-relevant information (e.g., product strategy, cross-functional leadership, user-centric methodologies)."
        "Maintain clarity and conciseness while aligning with the PM focus."
        "Always maintain accuracy and do not introduce non-factual information or unrealistic metrics."
        "Allow to technical skill be dominant"
        "Be concise"
        "\n\n"

    )

    # Define chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    chain = prompt | llm
    result = chain.invoke({"input": input_text})
    return result.content



