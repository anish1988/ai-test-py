import os 
from llama_index.core import StorageContext, VectorStoreIndex , load_index_from_storage
from llama_index.readers.file import PDFReader


def get_index(data, index_name):
    storage_context = StorageContext.from_defaults(persist_dir=index_name)
    if os.path.exists(index_name):
        index = load_index_from_storage(storage_context)
    else:
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True, storage_context=storage_context)
        index.set_index_id(index_name)
        index.storage_context.persist()
    return index
def get_pdf_index(pdf_path, index_name):
    reader = PDFReader()
    data = reader.load_data(pdf_path)
    index = get_index(data, index_name)
    return index
def get_pdf_index_from_path(pdf_path, index_name):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file {pdf_path} does not exist.")
    index = get_pdf_index(pdf_path, index_name)
    return index
def get_pdf_index_from_url(pdf_url, index_name):
    if not pdf_url.startswith("http"):
        raise ValueError(f"Invalid URL: {pdf_url}. URL must start with 'http'.")
    index = get_pdf_index(pdf_url, index_name)
    return index
def get_pdf_index_from_file(pdf_file, index_name):
    if not os.path.isfile(pdf_file):
        raise FileNotFoundError(f"PDF file {pdf_file} does not exist.")
    index = get_pdf_index(pdf_file, index_name)
    return index
def get_pdf_index_from_bytes(pdf_bytes, index_name):
    if not isinstance(pdf_bytes, bytes):
        raise ValueError(f"Invalid PDF bytes: {pdf_bytes}. PDF bytes must be of type 'bytes'.")
    index = get_pdf_index(pdf_bytes, index_name)
    return index
def get_pdf_index_from_string(pdf_string, index_name):
    if not isinstance(pdf_string, str):
        raise ValueError(f"Invalid PDF string: {pdf_string}. PDF string must be of type 'str'.")
    index = get_pdf_index(pdf_string, index_name)
    return index
def get_pdf_index_from_dict(pdf_dict, index_name):
    if not isinstance(pdf_dict, dict):
        raise ValueError(f"Invalid PDF dict: {pdf_dict}. PDF dict must be of type 'dict'.")
    index = get_pdf_index(pdf_dict, index_name)
    return index
def get_pdf_index_from_list(pdf_list, index_name):
    if not isinstance(pdf_list, list):
        raise ValueError(f"Invalid PDF list: {pdf_list}. PDF list must be of type 'list'.")
    index = get_pdf_index(pdf_list, index_name)
    return index
def get_pdf_index_from_tuple(pdf_tuple, index_name):
    if not isinstance(pdf_tuple, tuple):
        raise ValueError(f"Invalid PDF tuple: {pdf_tuple}. PDF tuple must be of type 'tuple'.")
    index = get_pdf_index(pdf_tuple, index_name)
    return index
def get_pdf_index_from_set(pdf_set, index_name):
    if not isinstance(pdf_set, set):
        raise ValueError(f"Invalid PDF set: {pdf_set}. PDF set must be of type 'set'.")
    index = get_pdf_index(pdf_set, index_name)
    return index
def get_pdf_index_from_frozenset(pdf_frozenset, index_name):
    if not isinstance(pdf_frozenset, frozenset):
        raise ValueError(f"Invalid PDF frozenset: {pdf_frozenset}. PDF frozenset must be of type 'frozenset'.")
    index = get_pdf_index(pdf_frozenset, index_name)
    return index
def get_pdf_index_from_bytearray(pdf_bytearray, index_name):
    if not isinstance(pdf_bytearray, bytearray):
        raise ValueError(f"Invalid PDF bytearray: {pdf_bytearray}. PDF bytearray must be of type 'bytearray'.")
    index = get_pdf_index(pdf_bytearray, index_name)
    return index
def get_pdf_index_from_memoryview(pdf_memoryview, index_name):
    if not isinstance(pdf_memoryview, memoryview):
        raise ValueError(f"Invalid PDF memoryview: {pdf_memoryview}. PDF memoryview must be of type 'memoryview'.")
    index = get_pdf_index(pdf_memoryview, index_name)
    return index
def get_pdf_index_from_array(pdf_array, index_name):
    if not isinstance(pdf_array, list):
        raise ValueError(f"Invalid PDF array: {pdf_array}. PDF array must be of type 'list'.")
    index = get_pdf_index(pdf_array, index_name)
    return index
def get_pdf_index_from_dataframe(pdf_dataframe, index_name):
    if not isinstance(pdf_dataframe, pd.DataFrame):
        raise ValueError(f"Invalid PDF dataframe: {pdf_dataframe}. PDF dataframe must be of type 'pd.DataFrame'.")
    index = get_pdf_index(pdf_dataframe, index_name)
    return index
def get_pdf_index_from_series(pdf_series, index_name):
    if not isinstance(pdf_series, pd.Series):
        raise ValueError(f"Invalid PDF series: {pdf_series}. PDF series must be of type 'pd.Series'.")
    index = get_pdf_index(pdf_series, index_name)
    return index  

pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file= pdf_path)
canada_index = get_index(canada_pdf, "canada index")
canada_engine = canada_index.as_query_engine()


      