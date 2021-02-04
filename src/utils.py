import re


def text_normalization(text):
    text = text.strip()
    # print('raw:', text)
    text = re.sub(r'([?.,!+-])', r' \1 ', text)
    # print('특수기호:', text)
    text = re.sub(r'[" "]+', " ", text)

    text = re.sub(r'[^0-9a-zA-Z가-힣?.,!+-]+', ' ', text)
    text = text.strip()
    # print('최종:', text)
    return text


if __name__ == '__main__':
    sample = '확인해 드릴게요, 세금을 포함해서 102만 원이라고 나오네요.'
    text_normalization(sample)
