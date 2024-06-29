import os
import traceback

from PIL import Image
from llama_index.core import ServiceContext, SimpleDirectoryReader, StorageContext, Settings
from llama_index.legacy.embeddings import HuggingFaceEmbedding

from multi_modal_lndex.base import MultiModalVectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.schema import ImageNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
# Create a local Qdrant vector store
import qdrant_client

def retrieve(retriever_engine, query_str):
    retrieval_results = retriever_engine.retrieve(query_str)
    retrieved_image = []
    retrieved_text = []

    for res_node in retrieval_results:
        print(res_node.score)
        if isinstance(res_node.node, ImageNode):
            print(res_node.node.metadata["file_name_img"])
            if res_node.score > 0.09:
                retrieved_image.append(res_node.node.metadata["file_path"])
                if "file_name_text" in res_node.node.metadata.keys():
                    with open(res_node.node.metadata["file_name_text"], "r", encoding="utf-8") as f:
                        retrieved_text.append(f.read())
            break
        else:
            if res_node.score > 0.5:
                retrieved_text.append(res_node.text)
                if "file_name_img" in res_node.node.metadata.keys():
                    retrieved_image.append(res_node.node.metadata["file_name_img"])
            break
    return retrieved_image, retrieved_text

def retrieve_image_to_image(retriever_engine, image):
    retrieval_results = retriever_engine.image_to_image_retrieve(image)
    retrieved_image = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_image.append(res_node.node.metadata["file_path"])
    return retrieved_image

def get_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_retriever_engine_from_local():
    try:
        client = qdrant_client.QdrantClient(path="qdrant_mm_db")
        Settings.llm = None
        Settings.embed_model = None
        text_store = QdrantVectorStore(
            client=client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=client, collection_name="image_collection"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )
        index = MultiModalVectorStoreIndex.from_vector_store(
            image_embed_model="siglip",
            storage_context=storage_context, service_context=None
        )
        retriever_engine = index.as_retriever(
            similarity_top_k=3, image_similarity_top_k=3
        )

    except Exception as e:
        traceback.print_exc()
        return 201, e
    return 200, {"retriever_engine":retriever_engine, "client": client, "index":index}


def update_nodes(file_path, image_folder, text_folder, cli_json):
    file_list = [file_path]
    client = cli_json["client"]
    index = cli_json["index"]
    try:
        documents = SimpleDirectoryReader(input_files=file_list).load_data()
        if image_folder is not None and text_folder is not None:
            for item in range(len(documents)):
                dir_file_img = os.listdir(f"storage/decompress/{image_folder}")
                dir_file_text = os.listdir(f"storage/decompress/{text_folder}")
                img_lastfix = dir_file_img[-1].split(".")[-1]
                text_lastfix = dir_file_text[-1].split(".")[-1]
                documents[item].metadata['file_name_img'] = file_list[item].replace(text_folder, image_folder).replace(text_lastfix,img_lastfix)
                documents[item].metadata['file_name_text'] = file_list[item].replace(image_folder, text_folder).replace(img_lastfix,text_lastfix)

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)
        index.insert_nodes(nodes=nodes)
        retriever_engine = index.as_retriever(
            similarity_top_k=3, image_similarity_top_k=3
        )
    except Exception as e:
        client.close()
        return 201, e
    return 200, {"retriever_engine":retriever_engine, "client": client, "index":index}

def get_retriever_engine(path, image_folder, text_folder):
    file_list = get_all_files(path)
    try:
        client = qdrant_client.QdrantClient(path="qdrant_mm_db")
        Settings.llm = None
        Settings.embed_model = None
        documents = SimpleDirectoryReader(input_files=file_list).load_data()

        if image_folder is not None and text_folder is not None:
            for item in range(len(documents)):
                dir_file_img = os.listdir(f"storage/decompress/{image_folder}")
                dir_file_text = os.listdir(f"storage/decompress/{text_folder}")
                img_lastfix = dir_file_img[-1].split(".")[-1]
                text_lastfix = dir_file_text[-1].split(".")[-1]
                documents[item].metadata['file_name_img'] = file_list[item].replace(text_folder, image_folder).replace(text_lastfix,img_lastfix)
                documents[item].metadata['file_name_text'] = file_list[item].replace(image_folder, text_folder).replace(img_lastfix,text_lastfix)

        parser = SentenceSplitter()
        nodes = parser.get_nodes_from_documents(documents)

        text_store = QdrantVectorStore(
            client=client, collection_name="text_collection"
        )
        image_store = QdrantVectorStore(
            client=client, collection_name="image_collection"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=text_store, image_store=image_store
        )
        index = MultiModalVectorStoreIndex(
            nodes=nodes,image_embed_model="siglip",
            storage_context=storage_context,service_context=None
        )

        retriever_engine = index.as_retriever(
            similarity_top_k=3, image_similarity_top_k=3
        )

    except Exception as e:
        client.close()
        return 201, e
    return 200, {"retriever_engine":retriever_engine, "client": client,"index":index}



