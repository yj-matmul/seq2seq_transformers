import re


def text_normalization(text):
    text = text.strip()
    print('raw:', text)
    text = re.sub(r'([?.,!+-])', r' \1 ', text)
    print('특수기호:', text)
    text = re.sub(r'[" "]+', " ", text)

    text = re.sub(r'[a-zA-Z가-힣?.,!+-]+', ' ', text)
    text = text.strip()
    return text
