import re
import logging
import torch.nn as nn
import bitsandbytes as bnb

LOGGER = logging.getLogger()


def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
    

def find_linear_names(model, train_mode = 'lora'):
    """
    This function identifies all linear layer names within a model that use 4-bit quantization.
    Args:
        model (torch.nn.Module): The PyTorch model to inspect.
    Returns:
        list: A list containing the names of all identified linear layers with 4-bit quantization.
    """
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear

    # Set to store identified layer names
    lora_module_names = set()

    # Iterate through named modules in the model
    for name, module in model.named_modules():
        # Check if the current module is an instance of the 4-bit linear layer class
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # Special case: remove 'lm_head' if present
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
    return list(lora_module_names)


class WikipediaTextPreprocessor:
    def __init__(self):
        pass
    
    def reduce_punctuation(self, text):
        return re.sub(r'\.{2,}', '.', text)
    
    def add_space_after_period(self, text):
        return re.sub(r'\.(\S)', r'. \1', text)
    
    def remove_dates(self, text):
        text = re.sub(r'날짜=\d{4}-\d{2}-\d{2}(\|)?', '', text)  # '날짜=YYYY-MM-DD' 형식만 제거
        text = re.sub(r'date=[a-zA-Z가-힣\s]*\d+', '', text)  # 'date=' 형식만 제거
        return text
    
    def remove_text_equals(self, text):
        return re.sub(r'text=', '', text)
        
    def remove_citations(self, text):
        return re.sub(r'p=\d+([–-]\d+)?|pp=\d+([–-]\d+)?|p=not cited|pp=not cited|page=\d+|pages=\d+[–-]\d+', '', text)
    
    def remove_group_and_name(self, text):
        return re.sub(r'group=\w+\||name=\w+\|', '', text)
    
    def remove_link_yes(self, text):
        return re.sub(r'\|link=yes', '', text)
    
    def remove_thumbnail(self, text):
        return re.sub(r'섬네일\|.*?\|.*?\|.*?(\n|$)', '', text, flags=re.DOTALL)
    
    def remove_date_with_text(self, text):
        return re.sub(r'\|.*?\d{4}년 \d{1,2}월 \d{1,2}일', '', text)
    
    def remove_date_with_text_and_suffix(self, text):
        return re.sub(r'\|.*?\d{4}년 \d{1,2}월 \d{1,2}일자', '', text)
    
    def remove_broken_html_refs(self, text):
        text = re.sub(r'<ref.*?>.*?(</ref>|</REF>|(\n|$))', '', text, flags=re.DOTALL)
        return re.sub(r'ref name="cc1"\/>', '', text)
    
    def remove_orphan_closing_refs(self, text):
        return re.sub(r'([.?!])[^.?!]*?</ref>', r'\1', text)
    
    def remove_order_and_st(self, text):
        text = re.sub(r'order=t', '', text)
        return re.sub(r'\(s=.*?\|t=.*?\)', '', text)
    
    def remove_pipe_number_pipe(self, text):
        return re.sub(r'\|\d+\|', '', text)
    
    def remove_title_until_newline(self, text):
        return re.sub(r'제목=.*?(\n|$)', '', text)
    
    def remove_quote_until_newline(self, text):
        return re.sub(r'인용구=.*?(\n|$)', '', text)
    
    def remove_description(self, text):
        return re.sub(r'\|설명=.*?:', '', text)
    
    def replace_newline_with_space(self, text):
        return text.replace('\\n', ' ').replace('\n', ' ')
    
    def reduce_multiple_spaces(self, text):
        return re.sub(r'\s{2,}', ' ', text)

    def preprocess_pipeline(self, text):
        text = self.reduce_punctuation(text)
        # text = self.add_space_after_period(text)
        text = self.remove_dates(text)
        text = self.remove_text_equals(text)
        text = self.remove_citations(text)
        text = self.remove_group_and_name(text)
        text = self.remove_link_yes(text)
        text = self.remove_thumbnail(text)
        # text = self.remove_date_with_text(text)
        # text = self.remove_date_with_text_and_suffix(text)
        text = self.remove_broken_html_refs(text)
        text = self.remove_orphan_closing_refs(text)
        text = self.remove_order_and_st(text)
        text = self.remove_pipe_number_pipe(text)
        text = self.remove_title_until_newline(text)
        text = self.remove_quote_until_newline(text)
        text = self.remove_description(text)
        text = self.replace_newline_with_space(text)
        text = self.reduce_multiple_spaces(text)
        
        return text 

text = "미국 상의원 또는 미국 상원(United States Senate)은 양원제인 미국 의회의 상원이다.\n\n미국 부통령이 상원의장이 된다. 각 주당 2명의 상원의원이 선출되어 100명의 상원의원으로 구성되어 있다. 임기는 6년이며, 2년마다 50개주 중 1/3씩 상원의원을 새로 선출하여 연방에 보낸다.\n\n미국 상원은 미국 하원과는 다르게 미국 대통령을 수반으로 하는 미국 연방 행정부에 각종 동의를 하는 기관이다. 하원이 세금과 경제에 대한 권한, 대통령을 포함한 대다수의 공무원을 파면할 권한을 갖고 있는 국민을 대표하는 기관인 반면 상원은 미국의 주를 대표한다. 즉 캘리포니아주, 일리노이주 같이 주 정부와 주 의회를 대표하는 기관이다. 그로 인하여 군대의 파병, 관료의 임명에 대한 동의, 외국 조약에 대한 승인 등 신속을 요하는 권한은 모두 상원에게만 있다. 그리고 하원에 대한 견제 역할(하원의 법안을 거부할 권한 등)을 담당한다. 2년의 임기로 인하여 급진적일 수밖에 없는 하원은 지나치게 급진적인 법안을 만들기 쉽다. 대표적인 예로 건강보험 개혁 당시 하원이 미국 연방 행정부에게 퍼블릭 옵션(공공건강보험기관)의 조항이 있는 반면 상원의 경우 하원안이 지나치게 세금이 많이 든다는 이유로 퍼블릭 옵션 조항을 제외하고 비영리건강보험기관이나 보험회사가 담당하도록 한 것이다. 이 경우처럼 상원은 하원이나 내각책임제가 빠지기 쉬운 국가들의 국회처럼 걸핏하면 발생하는 의회의 비정상적인 사태를 방지하는 기관이다. 상원은 급박한 처리사항의 경우가 아니면 법안을 먼저 내는 경우가 드물고 하원이 만든 법안을 수정하여 다시 하원에 되돌려보낸다. 이러한 방식으로 단원제가 빠지기 쉬운 함정을 미리 방지하는 것이다.날짜=2017-02-05"
preprocessor = WikipediaTextPreprocessor()
cleaned_text = preprocessor.preprocess_pipeline(text)
print(cleaned_text)