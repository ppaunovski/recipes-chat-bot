import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import json
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.vectorstores import VectorStoreRetriever

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import Qdrant

from langchain_core.runnables import Runnable


load_dotenv()
client = QdrantClient(url="http://localhost:6333")
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RecipesDocumentLoader(BaseLoader):

    def __init__(self, recipes: dict[str]) -> None:
        super().__init__()

        self.recipes = recipes

    def load(self) -> list[Document]:
        return [
            Document(
                recipe["recipe"],
                metadata={"title": recipe["title"], "ingrs": recipe["ingrs"]},
            )
            for recipe in self.recipes
        ]


def transform_dict_to_string(data: dict) -> str:
    title = data["title"]
    ingrs = data["ingredients"]
    instructions = data["instructions"]

    title_string = "Title: " + title
    instruction_string = "Instructions: \n" + "".join(
        sent + "\n" for sent in instructions
    )

    ingr_as_str = "Ingredients: \n" + "".join(
        f"{ingr['quantity']} {ingr['unit']} {ingr['name']}\n" for ingr in ingrs
    )

    return f"{title_string}\n{ingr_as_str}\n{instruction_string}"


def load_data_from_json(path_to_json: str) -> list[dict]:

    with open(path_to_json, "r") as file:
        data = json.load(file)

    return data


def create_dict_from_recipe(recipe: dict) -> dict:

    recipe_dict = {}
    recipe_dict["recipe"] = transform_dict_to_string(recipe)
    recipe_dict["title"] = recipe["title"]
    recipe_dict["ingrs"] = recipe["ingredients"]

    return recipe_dict


def get_recipes_data() -> list[dict]:
    data = load_data_from_json("data/other/1k.json")
    list_of_recipes = [create_dict_from_recipe(recipe) for recipe in data[:50]]
    return list_of_recipes


def load_recipes() -> list[Document]:
    list_of_recipes = get_recipes_data()
    document_loader = RecipesDocumentLoader(list_of_recipes)
    docs = document_loader.load()
    return docs


def get_chain(retriever: VectorStoreRetriever) -> Runnable:

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain


def query_to_answer(query: str) -> str:
    vector_store = Qdrant(
        client=client, collection_name="recipe_collection_1", embeddings=embeddings
    )

    if client.count("recipe_collection_1") == 0:
        vector_store.add_documents(documents=load_recipes())

    retriever = vector_store.as_retriever()

    chain = get_chain(retriever=retriever)
    response = chain.invoke({"input": query})

    return response["answer"]


if __name__ == "__main__":

    client.recreate_collection(
        collection_name="recipe_collection_1",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        timeout=30,
    )
