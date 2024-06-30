import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import json


class RecipesDocumentLoader(BaseLoader):

    def __init__(self, recipes: dict[str]) -> None:
        super().__init__()

        self.recipes = recipes


    def load(self) -> list[Document]:
        return [Document(recipe['recipe'],metadata={'title':recipe['title'], 'ingrs': recipe['ingrs']}) for recipe in self.recipes]
    


def transform_dict_to_string(data: dict) -> str:
    title = data['title']
    ingrs = data['ingredients']
    instructions = data['instructions']

    title_string = "Title: " + title
    instruction_string = "Instructions: \n" + "".join(sent + '\n' for sent in instructions)

    ingr_as_str = "Ingredients: \n" + "".join(f"{ingr['quantity']} {ingr['unit']} {ingr['name']}\n" for ingr in ingrs)

    return f'{title_string}\n{ingr_as_str}\n{instruction_string}'



def load_data_from_json(path_to_json: str) ->list[dict]:
    
    with open(path_to_json, 'r') as file:
        data = json.load(file)

    return data


def create_dict_from_recipe(recipe: dict) -> dict:
    recipe_dict = {}
    recipe_dict['recipe'] = transform_dict_to_string(recipe)
    recipe_dict['title'] = recipe['title']
    recipe_dict['ingrs'] = recipe['ingredients']

    return recipe_dict


def get_recipes_data() -> None:
    data = load_data_from_json('1_k.sjoon')
    list_of_recipes = [create_dict_from_recipe(recipe) for recipe in data[:50]]
    return list_of_recipes


def load_reciples():
    document_loader = RecipesDocumentLoader(list_of_recipes)

# Load the documents
    docs = document_loader.load(    )


def get_chain(query):
    combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Example query
    query = "Give me a mac and cheese recipe"

# Run the RAG pipeline
    response = retrieval_chain.invoke({"input": query})

    return response