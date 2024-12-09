import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_text_splitters import NLTKTextSplitter
from langchain_text_splitters import KonlpyTextSplitter
import random
import io
import pandas as pd

random.seed(1004)

menu = st.sidebar.radio("Menu", ["langchain corpus split", "EDA plot"])

if menu == 'langchain corpus split':
    st.title("langchain corpus -> phrase")

    # 데이터프레임 선택
    selected_dataframe = st.selectbox(
        "Select a DataFrame for splitting",
        ["train_df", "valid_df", "corpus_df"]
    )

    # 선택한 데이터프레임에 따라 CSVLoader로 데이터를 불러오기
    if selected_dataframe == "train_df":
        loader = CSVLoader(file_path='dataset/train.csv', encoding='utf-8', source_column='context')
    elif selected_dataframe == "valid_df":
        loader = CSVLoader(file_path='dataset/validation.csv', encoding='utf-8', source_column='context')
    elif selected_dataframe == "corpus_df":
        loader = CSVLoader(file_path='dataset/wikipedia_documents.csv', encoding='utf-8', source_column='text')


    # CSVLoader로 데이터 불러오기
    data = loader.load()

    st.write(f"Selected DataFrame: {selected_dataframe}")
    st.dataframe(pd.DataFrame([doc.page_content for doc in data]).head())  # 선택된 데이터프레임의 첫 5개 행을 출력

    # Splitter 선택
    selected_splitter = st.selectbox("Select a splitter", [
        'RecursiveCharacterTextSplitter',
        'CharacterTextSplitter',
        'TokenTextSplitter',
        'SentenceTransformersTokenTextSplitter',
        'NLTKTextSplitter',
        'KonlpyTextSplitter'])

    chunk_size = st.number_input("Enter chunk size", min_value=1, max_value=5000, value=500, step=1)
    chunk_overlap = st.number_input("Enter chunk overlap", 0, chunk_size, 100, step=1)

    st.divider()

    # 랜덤으로 5개의 문서 샘플 선택
    random_data_samples = random.sample(data, 5)

    if selected_splitter == 'RecursiveCharacterTextSplitter':
        st.header('RecursiveCharacterTextSplitter')
        splitter = RecursiveCharacterTextSplitter(separators=['\n'], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif selected_splitter == 'CharacterTextSplitter':
        st.header('CharacterTextSplitter')
        splitter = CharacterTextSplitter(separator=' ',chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif selected_splitter == 'TokenTextSplitter':
        st.header('TokenTextSplitter')
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif selected_splitter == 'SentenceTransformersTokenTextSplitter':
        st.header('SentenceTransformersTokenTextSplitter')
        splitter = SentenceTransformersTokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif selected_splitter == 'NLTKTextSplitter':
        st.header('NLTKTextSplitter')
        splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    elif selected_splitter == 'KonlpyTextSplitter':
        st.header('KonlpyTextSplitter')
        splitter = KonlpyTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 문서 분리 작업 (랜덤 샘플)
    docs_by_document = []
    for doc in random_data_samples:
        split_chunks = splitter.split_text(doc.page_content)
        docs_by_document.append((doc, split_chunks))  # 원래 문서와 분리된 chunk들을 함께 저장

    # 생성된 chunk 개수 출력
    st.subheader(f'Created total chunks: {sum(len(chunks) for _, chunks in docs_by_document)}')

    # 선택된 5개의 문서 각각의 전체 길이 출력
    for idx, (doc, chunks) in enumerate(docs_by_document):
        st.header(f'Document {idx + 1} total length: {len(doc.page_content)}')
        st.subheader(f'Document {idx + 1}, Chunks amount: {len(chunks)}ea')
        for chunk_idx, chunk in enumerate(chunks):
            st.write(f'Chunk {chunk_idx + 1} content length: {len(chunk)}')
            st.write(chunk)
            st.divider()

# 전체 데이터를 처리하여 CSV 파일 저장
    if st.button('Splitted context save'):
        with st.spinner('Splitting and saving data...'):
            context_save = []

            # 모든 문서를 분리
            for idx, doc in enumerate(data):
                split_chunks = splitter.split_text(doc.page_content)
                for chunk_idx, chunk in enumerate(split_chunks):
                    context_save.append({
                        'document_id': idx,  # 원본 document ID
                        'chunk_index': chunk_idx,  # 분리된 청크 인덱스
                        'chunk_content': chunk  # 청크 내용
                    })

            # 분리된 데이터를 데이터프레임으로 변환
            split_df = pd.DataFrame(context_save)

            csv_buffer = io.BytesIO()
            split_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_buffer.seek(0)

            st.download_button(
                label=f"Download CSV_({selected_dataframe})",
                data=csv_buffer,
                file_name=f"{selected_dataframe}_splitted_documents.csv",
                mime="text/csv"
            )

else:
    st.title("EDA plot")
