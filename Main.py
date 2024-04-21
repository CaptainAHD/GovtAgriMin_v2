import streamlit as st
import os
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from openai import OpenAI

st.set_page_config(layout="wide")

st.title('Crop Data Search System')
# os.environ["OPENAI_API_KEY"] = ""
# API_KEY = ""
client = OpenAI()

@st.cache_resource
def create_retriever(top_k):
    index = load_index_from_storage(
        storage_context=StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir="vector store"),
            vector_store=FaissVectorStore.from_persist_dir(persist_dir="vector store"),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir="vector store"),
        )
    )
    return index.as_retriever(retriever_mode="embedding", similarity_top_k=int(top_k))

st.title("Crop Data Semantic Search")

query = st.text_input(label="Please enter your query - ", value="")
top_k = st.number_input(label="Top k - ", min_value=3, max_value=5, value=3)

retriever = create_retriever(top_k)

if query and top_k:
    col1, col2 = st.columns([3, 2])
    with col1:
        response = []
        for i in retriever.retrieve(query):
            response.append(
                {
                    "Document": i.metadata["link"][40:-4],
                    "Source": i.metadata["link"],
                    "Text": i.get_text(),
                    "Score": i.get_score(),
                }
            )
        st.json(response)

    with col2:
        summary = st.empty()
        top3 = []
        top3_couplet = []
        top3_name = []
        for i in response:
            top3.append(i["Text"])
            top3_name.append(i["Document"])
        temp_summary = []
        for resp in client.chat.completions.create(
            model="gpt-4-1106-preview",
            # response_format="json_object",
            messages=[
                {
                    "role": "system",
                    "content": f"Act as a query answering GPT for The Ministry of Agriculture and Farmers Welfare, India. You answer queries of officers and farmers using your knowledgebase. Now answer the {query}, using the following knowledgebase:{top3} Your knowledgebase also contains name of the document, give it when answering so as to making your answer clear: {top3_name}. Strictly answer based on the available knowledge base.",
                },
                {
                    "role": "user",
                    "content": f"""Summarize the following interpretation of couplets in context of the query “{query}”:

{top3_name[2]}
Summary:
{top3[2]}

{top3_name[1]}
Summary:
{top3[1]}

{top3_name[0]}
Summary:
{top3[0]}""",
                },
            ],
            stream=True,
        ):
            if resp.choices[0].finish_reason == "stop":
                break
            temp_summary.append(resp.choices[0].delta.content)
            result = "".join(temp_summary).strip()
            # Revised replacement logic for hyperlinks
            for phrase, link in {
                "Thrips": "https://drive.google.com/file/d/1Tnps02E_hBCgrdiS3etVV_J3hjT0xEyf/view?usp=share_link",
                "Whitefly": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
                "whiteflies": "https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link",
                "PBW": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Pink Bollworm": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Cotton PBW Larva": "https://drive.google.com/file/d/1l8HOlfZNbce_qHbaZujXO4KB_ug_SZZ3/view?usp=share_link",
                "Cotton Whitefly damage symptom": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "Cotton Whitefly damage symptoms": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "damage symptoms of Cotton Whitefly": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "Whitefly damage symptoms": "https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link",
                "Fall Army worm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
                "Fall Armyworm": "https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link",
                "FF adult on Mango": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
                "FF damage to Indian crops": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
                "fruit flies on mangoes": "https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link",
                "FF Egg laying": "https://drive.google.com/file/d/1BVaNTtlG9Y7nSiOUqAS7yhfVnjDLPMkr/view?usp=share_link",
                "FF fruit damage": "https://drive.google.com/file/d/1oSRuO3M2D1wfiTPqA9VSxzSgambN7BXF/view?usp=share_link",
                "FF Larve damage": "https://drive.google.com/file/d/1Nr_ZwQEAIlgWoNjIEuXW_LG5yu_s7eHT/view?usp=share_link",
                "FF Oozing": "https://drive.google.com/file/d/1Sht1JZGlg_SqUWo0rN1stPL1FGqUYGtZ/view?usp=share_link",
                "FF Puncture": "https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link",
                "Fruit Fly": "https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link",
                "Fruitfly in Mango fruit": "https://drive.google.com/file/d/16zarIaupOIWAK2GrpBmQy214MqLfML53/view?usp=share_link",
                "Fruitfly in Mango leaf": "https://drive.google.com/file/d/1de4XhE1RQ5GKvOZcmkoz9Yi-q1JlQo1w/view?usp=share_link",
                "bore hole caused by larva of Yellow stem borer": "https://drive.google.com/file/d/1guo1cO2f1IjRPztTiZS9OemjFLq8KTiP/view?usp=share_link",
                "bore hole of YSB larva": "https://drive.google.com/file/d/1_k0msr8JRUp5uUUKh5dVBHepLNzb-Oyd/view?usp=share_link",
                "larva of Yellow stem borer": "https://drive.google.com/file/d/1L9WOrmqUPOUrzib17USsXrBWU6EcgYxX/view?usp=share_link",
                "Moth of Yellow Stem borer on paddy crop": "https://drive.google.com/file/d/12J37UHo_P5zPWAn4zU3zw35nDdzsIp1K/view?usp=share_link",
                "Moth of Yellow Stem borer": "https://drive.google.com/file/d/1St9fNNmMy1Sy_p_W6hTtjqhHF2UbGyfb/view?usp=share_link",
                "RICE- Yellow stem borer- Scirpophaga incertulas": "https://drive.google.com/file/d/1dw5hlAwPQFk5WodHbY72FkWwLDmdmCMr/view?usp=share_link",
                "Cotton PBW Damage Symptom": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "symptoms for Pink Bollworm (PBW) damage": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "symptoms of Cotton Whitefly damage": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Cotton PBW (Pink Bollworm) Damage Symptoms": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link",
                "Cotton PBW (Pink Bollworm) Damage Symptom": "https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link"
            }.items():
                result = result.replace(phrase, f'<a href="{link}" style="text-decoration: underline;">{phrase}</a>')

            # # Display the result
            summary.markdown(result, unsafe_allow_html=True)














# import streamlit as st
# import os
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.core import load_index_from_storage, StorageContext
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core.storage.index_store import SimpleIndexStore
# from openai import OpenAI

# st.set_page_config(layout="wide")


# API_KEY = " "
# client = OpenAI(api_key=API_KEY)


# @st.cache_resource
# def create_retriever():
#     index = load_index_from_storage(
#         storage_context=StorageContext.from_defaults(
#             docstore=SimpleDocumentStore.from_persist_dir(persist_dir="vector store"),
#             vector_store=FaissVectorStore.from_persist_dir(persist_dir="vector store"),
#             index_store=SimpleIndexStore.from_persist_dir(persist_dir="vector store"),
#         )
#     )
#     # print(index)
#     return index.as_retriever(retriever_mode="embedding", similarity_top_k=int(top_k))


# st.title("Crop Data Semantic Search")

# query = st.text_input(label="Please enter your query - ", value="")
# top_k = st.number_input(label="Top k - ", min_value=3, max_value=5, value=3)

# retriever = create_retriever()

# if query and top_k:
#     col1, col2 = st.columns([3, 2])
#     with col1:
#         response = []
#         for i in retriever.retrieve(query):
#             response.append(
#                 {
#                     "Document": i.metadata["link"][40:-4],
#                     "Source": i.metadata["link"],
#                     "Text": i.get_text(),
#                     "Score": i.get_score(),
#                 }
#             )

#         # Display response
#         # Display response
#         for i in response:
#             text_with_links = i["Text"]  # Get the text
#             # Add hyperlinks to specific words or phrases
#             text_with_links = text_with_links.replace(
#                 "Thrips",
#                 f"<u><a href='https://drive.google.com/file/d/1Tnps02E_hBCgrdiS3etVV_J3hjT0xEyf/view?usp=share_link'>Thrips</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Whitefly",
#                 f"<u><a href='https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link'>Whitefly</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "PBW",
#                 f"<u><a href='https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link'>PBW</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Cotton PBW Damage Symptom",
#                 f"<u><a href='https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link'>Cotton PBW  Damage Symptom</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Cotton PBW Larva",
#                 f"<u><a href='https://drive.google.com/file/d/1l8HOlfZNbce_qHbaZujXO4KB_ug_SZZ3/view?usp=share_link'>Cotton PBW  Larva</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Cotton Whitefly damage symptom",
#                 f"<u><a href='https://drive.google.com/file/d/1o9NIiU0nEHDQF6t0fnIuNgv1suFpUME7/view?usp=share_link'>Cotton Whitefly  damage symptom</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Fall Army worm",
#                 f"<u><a href='https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link'>Fall Army worm</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "FF adult on Mango",
#                 f"<u><a href='https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link'>FF adult on Mango</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "FF Egg laying",
#                 f"<u><a href='https://drive.google.com/file/d/1BVaNTtlG9Y7nSiOUqAS7yhfVnjDLPMkr/view?usp=share_link'>FF Egg laying</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "FF fruit damage",
#                 f"<u><a href='https://drive.google.com/file/d/1oSRuO3M2D1wfiTPqA9VSxzSgambN7BXF/view?usp=share_link'>FF fruit damage</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "FF Larve damage",
#                 f"<u><a href='https://drive.google.com/file/d/1Nr_ZwQEAIlgWoNjIEuXW_LG5yu_s7eHT/view?usp=share_link'>FF Larve damage</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "FF Oozing",
#                 f"<u><a href='https://drive.google.com/file/d/1Sht1JZGlg_SqUWo0rN1stPL1FGqUYGtZ/view?usp=share_link'>FF Oozing</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "FF Puncture",
#                 f"<u><a href='https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link'>FF Puncture</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Fruit Fly",
#                 f"<u><a href='https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link'>Fruit Fly</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Fruitfly in Mango fruit",
#                 f"<u><a href='https://drive.google.com/file/d/16zarIaupOIWAK2GrpBmQy214MqLfML53/view?usp=share_link'>Fruitfly in Mango fruit</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Fruitfly in Mango leaf",
#                 f"<u><a href='https://drive.google.com/file/d/1de4XhE1RQ5GKvOZcmkoz9Yi-q1JlQo1w/view?usp=share_link'>Fruitfly in Mango leaf</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "bore hole caused by larva of Yellow stem borer",
#                 f"<u><a href='https://drive.google.com/file/d/1guo1cO2f1IjRPztTiZS9OemjFLq8KTiP/view?usp=share_link'>bore hole caused by larva of Yellow stem borer</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "bore hole of YSB larva",
#                 f"<u><a href='https://drive.google.com/file/d/1_k0msr8JRUp5uUUKh5dVBHepLNzb-Oyd/view?usp=share_link'>bore hole of YSB larva</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "larva of Yellow stem borer",
#                 f"<u><a href='https://drive.google.com/file/d/1L9WOrmqUPOUrzib17USsXrBWU6EcgYxX/view?usp=share_link'>larva of Yellow stem borer</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Moth of Yellow Stem borer on paddy crop",
#                 f"<u><a href='https://drive.google.com/file/d/12J37UHo_P5zPWAn4zU3zw35nDdzsIp1K/view?usp=share_link'>Moth of Yellow Stem borer on paddy crop</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "Moth of Yellow Stem borer",
#                 f"<u><a href='https://drive.google.com/file/d/1St9fNNmMy1Sy_p_W6hTtjqhHF2UbGyfb/view?usp=share_link'>Moth of Yellow Stem borer</a></u>",
#             )
#             text_with_links = text_with_links.replace(
#                 "RICE- Yellow stem borer- Scirpophaga incertulas",
#                 f"<u><a href='https://drive.google.com/file/d/1dw5hlAwPQFk5WodHbY72FkWwLDmdmCMr/view?usp=share_link'>RICE- Yellow stem borer- Scirpophaga incertulas</a></u>",
#             )
#             # Add more links and names as needed
#             st.markdown(text_with_links, unsafe_allow_html=True)

#             # Thrips symptom on leaf  -  https://drive.google.com/file/d/1Tnps02E_hBCgrdiS3etVV_J3hjT0xEyf/view?usp=share_link
#             # Cotton Whitefly - damage symptom  -  https://drive.google.com/file/d/15GYYUISigHrHrsBgYAKpoZxA6r0iDrlA/view?usp=share_link
#             # Cotton PBW - damage Symptom  -  https://drive.google.com/file/d/1q4m7tiVgwD3NJynFYKmhrRbSKtJOwsVe/view?usp=share_link
#             # Fall Army worm  -  https://drive.google.com/file/d/1VxQ3IRVa78fIQE1sS8eLLQCaqLZQtZ2f/view?usp=share_link
#             # FAW egg mass  -  https://drive.google.com/file/d/1Jw1BLgsdRz_fXApfEu8HUrLsFr_u4s50/view?usp=share_link
#             # FAW larva  -  https://drive.google.com/file/d/17NGOC4nFiO_Rp6K66erOSBohYro-YB9c/view?usp=share_link
#             # Maize FAW  -  https://drive.google.com/file/d/1w7b3ruVilzSpU4EuDpl3ChyC3iSou5XL/view?usp=share_link
#             # FF adult on Mango  -  https://drive.google.com/file/d/11qedO5ek3yBkwcOabgSlHmFZWSDoyCo_/view?usp=share_link
#             # FF Egg laying  -  https://drive.google.com/file/d/1BVaNTtlG9Y7nSiOUqAS7yhfVnjDLPMkr/view?usp=share_link
#             # FF fruit damage  -  https://drive.google.com/file/d/1oSRuO3M2D1wfiTPqA9VSxzSgambN7BXF/view?usp=share_link
#             # FF Larve damage  -  https://drive.google.com/file/d/1Nr_ZwQEAIlgWoNjIEuXW_LG5yu_s7eHT/view?usp=share_link
#             # FF Oozing  -  https://drive.google.com/file/d/1Sht1JZGlg_SqUWo0rN1stPL1FGqUYGtZ/view?usp=share_link
#             # FF Puncture  -  https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link
#             # Fruit Fly  -  https://drive.google.com/file/d/1cBvmJFCmRveDTwiP6FEHO_leylieq9rR/view?usp=share_link
#             # Fruitfly in Mango fruit  -  https://drive.google.com/file/d/16zarIaupOIWAK2GrpBmQy214MqLfML53/view?usp=share_link
#             # Fruitfly in Mango leaf  -  https://drive.google.com/file/d/1de4XhE1RQ5GKvOZcmkoz9Yi-q1JlQo1w/view?usp=share_link
#             # bore hole caused by larva of Yellow stem borer  -  https://drive.google.com/file/d/1guo1cO2f1IjRPztTiZS9OemjFLq8KTiP/view?usp=share_link
#             # bore hole of YSB larva  -  https://drive.google.com/file/d/1_k0msr8JRUp5uUUKh5dVBHepLNzb-Oyd/view?usp=share_link
#             # larva of Yellow stem borer  -  https://drive.google.com/file/d/1L9WOrmqUPOUrzib17USsXrBWU6EcgYxX/view?usp=share_link
#             # Moth of Yellow Stem borer on paddy crop  -  https://drive.google.com/file/d/12J37UHo_P5zPWAn4zU3zw35nDdzsIp1K/view?usp=share_link
#             # Moth of Yellow Stem borer  -  https://drive.google.com/file/d/1St9fNNmMy1Sy_p_W6hTtjqhHF2UbGyfb/view?usp=share_link
#             # RICE- Yellow stem borer- Scirpophaga incertulas  -  https://drive.google.com/file/d/1dw5hlAwPQFk5WodHbY72FkWwLDmdmCMr/view?usp=share_link
#             st.markdown(text_with_links, unsafe_allow_html=True)

#     with col2:
#         summary = st.empty()
#         top3 = []
#         top3_couplet = []
#         top3_name = []
#         for i in response:
#             top3.append(i["Text"])
#             top3_name.append(i["Document"])
#         temp_summary = []
#         for resp in client.chat.completions.create(
#             model="gpt-4-1106-preview",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": f"Act as a query answering GPT for The Ministry of Agriculture and Farmers Welfare, India. You answer queries of officers and farmers using your knowledgebase. Now answer the {query}, using the following knowledgebase:{top3} Your knowledgebase also contains name of the document, give it when answering so as to making your answer clear: {top3_name}. Strictly answer based on the available knowledge base.",
#                 },
#                 {
#                     "role": "user",
#                     "content": f"""Summarize the following interpretation of couplets in context of the query “{query}”:

# {top3_name[2]}
# Summary:
# {top3[2]}

# {top3_name[1]}
# Summary:
# {top3[1]}

# {top3_name[0]}
# Summary:
# {top3[0]}""",
#                 },
#             ],
#             stream=True,
#         ):
#             if resp.choices[0].finish_reason == "stop":
#                 break
#             temp_summary.append(resp.choices[0].delta.content)
#             result = "".join(temp_summary).strip()
#             summary.markdown(f"{result}")


# import streamlit as st
# import os
# import re

# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.core import load_index_from_storage, StorageContext
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core.storage.index_store import SimpleIndexStore
# from openai import OpenAI

# # Set Streamlit page config
# st.set_page_config(layout="wide")

# # Set OpenAI API Key

# API_KEY = " "
# client = OpenAI(api_key=API_KEY)

# # Define a CSS class for consistent link styling
# st.markdown(
#     """
#     <style>
#     .hyperlink {
#         color: blue;
#         text-decoration: underline;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# # Function to create retriever
# @st.cache_resource
# def create_retriever(top_k):
#     st.write("Creating Retriever...")
#     index = load_index_from_storage(
#         storage_context=StorageContext.from_defaults(
#             docstore=SimpleDocumentStore.from_persist_dir(persist_dir="vector store"),
#             vector_store=FaissVectorStore.from_persist_dir(persist_dir="vector store"),
#             index_store=SimpleIndexStore.from_persist_dir(persist_dir="vector store"),
#         )
#     )
#     st.write("Retriever created successfully!")
#     return index.as_retriever(retriever_mode="embedding", similarity_top_k=int(top_k))


# # Function to identify relevant crop diseases
# def identify_crop_diseases(rag_text):
#     st.write("Identifying Crop Diseases...")
#     st.write(f"Text to identify crop diseases from: {rag_text}")  # Debug statement
#     pattern = r"R:\s*([A-Za-z\s]+),"
#     crop_diseases = re.findall(pattern, rag_text)
#     st.write(f"Crop Diseases Identified: {crop_diseases}")
#     return crop_diseases


# # Function to create hyperlinks for crop diseases
# def create_hyperlinks_styled(rag_text, crop_diseases, crop_disease_images):
#     st.write("Creating Hyperlinks...")
#     hyperlinked_rag = rag_text
#     for disease in crop_diseases:
#         image_url = crop_disease_images.get(disease, "Image Not Found")
#         hyperlink = f'<a href="{image_url}" class="hyperlink">{disease}</a>'
#         # Check if the disease is in the image dictionary and it's the first occurrence
#         if disease in crop_disease_images and hyperlinked_rag.count(disease) == 1:
#             hyperlinked_rag = hyperlinked_rag.replace(
#                 disease,
#                 f'<span style="text-decoration: underline; background-color: yellow;">{hyperlink}</span>',
#                 1,
#             )
#         else:
#             hyperlinked_rag = hyperlinked_rag.replace(disease, hyperlink)
#     st.write("Hyperlinks created successfully!")
#     return hyperlinked_rag


# # Main Streamlit app
# def main():
#     st.title("Crop Data Semantic Search")

#     query = st.text_input(label="Please enter your query - ", value="")
#     top_k = st.number_input(label="Top k - ", min_value=3, max_value=5, value=3)

#     retriever = create_retriever(top_k)

#     if query and top_k:
#         col1, col2 = st.columns([3, 2])
#         with col1:
#             response = []
#             for i in retriever.retrieve(query):
#                 response.append(
#                     {
#                         "Document": i.metadata["link"][40:-4],
#                         "Source": i.metadata["link"],
#                         "Text": i.get_text(),
#                         "Score": i.get_score(),
#                     }
#                 )
#             st.json(response)

#             if response:
#                 crop_diseases = identify_crop_diseases(response[0]["Text"])
#                 if not crop_diseases:
#                     st.write("No crop diseases identified!")
#                 else:
#                     images = {
#                         "Thrips": "https://drive.google.com/file/d/1JzlVr7lBuR2wmAb66CFT4969FWzqQxAd/view?usp=sharing",
#                         "PBW": "https://drive.google.com/file/d/17XysHMIeT_nn0JykEAG3LLabznmfkqBB/view?usp=sharing",
#                         "Whitefly": "https://drive.google.com/file/d/1azSnSiD52TRNi8PbHvs5A7pxD2w4MBoM/view?usp=sharing",
#                         # Add other diseases and their image URLs here
#                     }
#                     hyperlinked_rag = create_hyperlinks_styled(response[0]["Text"], crop_diseases, images)
#                     st.markdown(hyperlinked_rag, unsafe_allow_html=True)

#         with col2:
#             summary = st.empty()
#             top3 = []
#             top3_couplet = []
#             top3_name = []
#             for i in response:
#                 top3.append(i["Text"])
#                 top3_name.append(i["Document"])
#             temp_summary = []
#             for resp in client.chat.completions.create(
#                 model="gpt-4-1106-preview",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": f"Act as a query answering GPT for The Ministry of Agriculture and Farmers Welfare, India. You answer queries of officers and farmers using your knowledgebase. Now answer the {query}, using the following knowledgebase:{top3} Your knowledgebase also contains name of the document, give it when answering so as to making your answer clear: {top3_name}. Strictly answer based on the available knowledge base.",
#                     },
#                     {
#                         "role": "user",
#                         "content": f"""Summarize the following interpretation of couplets in context of the query “{query}”:

#                                                             {top3_name[2]}
#                                                             Summary:
#                                                             {top3[2]}

#                                                             {top3_name[1]}
#                                                             Summary:
#                                                             {top3[1]}

#                                                             {top3_name[0]}
#                                                             Summary:
#                                                             {top3[0]}""",
#                     },
#                 ],
#                 stream=True,
#             ):
#                 if resp.choices[0].finish_reason == "stop":
#                     break
#                 temp_summary.append(resp.choices[0].delta.content)
#                 result = "".join(temp_summary).strip()
#                 summary.markdown(f"{result}")


# if __name__ == "__main__":
#     main()


# import streamlit as st
# import os
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.core import load_index_from_storage, StorageContext
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core.storage.index_store import SimpleIndexStore
# from openai import OpenAI

# st.set_page_config(layout = "wide")

# # client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])
# # os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


# API_KEY = ' '
# client = OpenAI(api_key=API_KEY)


# @st.cache_resource
# def create_retriever():
#     index = load_index_from_storage(storage_context = StorageContext.from_defaults(
#                 docstore = SimpleDocumentStore.from_persist_dir(persist_dir = "vector store"),
#                 vector_store = FaissVectorStore.from_persist_dir(persist_dir = "vector store"),
#                 index_store = SimpleIndexStore.from_persist_dir(persist_dir = "vector store"),
#             ))
#     return index.as_retriever(retriever_mode = 'embedding', similarity_top_k = int(top_k))

# st.title('Crop Data Semantic Search')

# query = st.text_input(label = 'Please enter your query - ', value = '')
# top_k = st.number_input(label = 'Top k - ', min_value = 3, max_value = 5, value = 3)

# retriever = create_retriever()

# if query and top_k:
#     col1, col2 = st.columns([3, 2])
#     with col1:
#         response = []
#         for i in retriever.retrieve(query):
#             response.append({
#                     'Document' : i.metadata['link'][40:-4],
#                     'Source' : i.metadata['link'],
#                     'Text' : i.get_text(),
#                     'Score' : i.get_score(),
#                 })
#         st.json(response)

#     with col2:
#         summary = st.empty()
#         top3 = []
#         top3_couplet = []
#         top3_name = []
#         for i in response:
#              top3.append(i["Text"])
#              top3_name.append(i["Document"])
#         temp_summary = []
#         for resp in client.chat.completions.create(model = "gpt-4-1106-preview",
#             messages = [
#                     {"role": "system", "content": f"Act as a query answering GPT for The Ministry of Agriculture and Farmers Welfare, India. You answer queries of officers and farmers using your knowledgebase. Now answer the {query}, using the following knowledgebase:{top3} Your knowledgebase also contains name of the document, give it when answering so as to making your answer clear: {top3_name}. Strictly answer based on the available knowledge base."},
#                     {"role": "user", "content": f"""Summarize the following interpretation of couplets in context of the query “{query}”:

# {top3_name[2]}
# Summary:
# {top3[2]}

# {top3_name[1]}
# Summary:
# {top3[1]}

# {top3_name[0]}
# Summary:
# {top3[0]}"""},
#                 ],
#             stream = True):
#                 if resp.choices[0].finish_reason == "stop":
#                     break
#                 temp_summary.append(resp.choices[0].delta.content)
#                 result = "".join(temp_summary).strip()
#                 summary.markdown(f'{result}')
