from bs4 import BeautifulSoup
import pandas as pd
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pykospacing import Spacing


class TextCleaner:
    def __init__(self, df=None):
        self.df = df

    def remove_duplicates(self):
        if self.df is not None:
            self.df = self.df.drop_duplicates(subset=['title', 'text'])
        return self.df

    def remove_HTML(self):
        if self.df is not None and 'text' in self.df.columns:
            self.df['text'] = self.df['text'].apply(lambda x: BeautifulSoup(x, "html.parser").get_text() if isinstance(x, str) else x)
        return self.df

    def remove_annot(self):
        if self.df is not None and 'text' in self.df.columns:
            self.df['text'] = self.df['text'].apply(lambda x: re.sub(r'p{1,2}=\d+(–?\d+)?(,\s*\d+)*|loc\s*=\s*§?\s*\d*', '', x) if isinstance(x, str) else x)
        return self.df

    def remove_fields(self, fields):
        if self.df is not None and 'text' in self.df.columns:
            pattern = r'\b(' + '|'.join(fields) + r')\s*=\s*[^|]+(\||$)'
            self.df['text'] = self.df['text'].apply(lambda x: self._apply_field_removal(x, pattern) if isinstance(x, str) else x)
        return self.df

    def _apply_field_removal(self, text, pattern):
        # Removing the specified fields
        text = re.sub(pattern, '', text)

        # JSON 객체 제거
        text = re.sub(r'\{.*?\}', '', text)

        # URL 제거
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'http:\S+', '', text)

        # 첨부 파일 제거
        text = re.sub(r'파일:\S+\s*', '', text)

        # 위키 템플릿 제거
        text = re.sub(r'\|\s*\w+\s*=\s*[^|}]+', '', text)

        text = re.sub(r'[|]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def remove_non_korean_english_words(self):
        if self.df is not None and 'text' in self.df.columns:
            self.df['text'] = self.df['text'].apply(lambda x: self._apply_language_filter(x) if isinstance(x, str) else x)
        return self.df
    
    def _apply_language_filter(self, text):
        words = text.split()
        filtered_words = []

        for word in words:
            if re.fullmatch(r'[가-힣0-9]+', word):
                filtered_words.append(word)
                continue

            if re.fullmatch(r'[a-zA-Z0-9]+', word):
                try:
                    lang = detect(word)
                    if lang == 'en':
                        filtered_words.append(word)
                except LangDetectException:
                    continue
                continue

            cleaned_word = re.sub(r'[^\w\s]', '', word)
            if re.fullmatch(r'[가-힣a-zA-Z0-9]+', cleaned_word):
                filtered_words.append(cleaned_word)

        return ' '.join(filtered_words)

    def correct_space(self, spacing_function):
        if self.df is not None and 'text' in self.df.columns:
            self.df['text'] = self.df['text'].apply(lambda x: spacing_function(x) if isinstance(x, str) else x)
        return self.df