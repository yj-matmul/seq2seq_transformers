# BERT pytorch 버전에 쓰인 각종 문법
## BERT model 의 실행 구조
#### hugging face 코드를 보면 안쪽에서부터 class 구조를 만들고 있음
#### 위에서부터 순차적으로 봐도 무방

'''
1. tf checkpoint 을 pytorch checkpoint로 변환
    def load_tf_weights_in_bert

2. Bert의 기본 정보 규격 생성
    class BertConfig
        class BertLayerNorm 클래스 선언

3. BertModel 실행
    class BertModel(class BertPreTrainModel 상속)
        class BertEmbeddings
        class BertEncoder
            class BertLayer
                class BertAttention
                    class BertSelfAttention
                    class BertSelfOutput
                class BertIntermediate
                class BertOutput
        class BertPooler
'''

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

## __init__(생성자) 선언 시 보이는 super()
#### child class 에서 parent class 의 내용을 사용하고 싶을 경우에 이용

#### 오버라이딩 발생
class father():  # 부모 클래스
    def handsome(self):
        print("잘생겼다")


class brother(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    '''아들'''


class sister(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    def pretty(self):
        print("예쁘다")

    def handsome(self):
        '''물려받았어요'''


brother = brother()
brother.handsome()

girl = sister()
girl.handsome()  # 오버라이딩으로 실행 내용이 수정돼 출력 내용 없음
girl.pretty()

#### super로 parent class method 이용
class father():  # 부모 클래스
    def handsome(self):
        print("잘생겼다")


class brother(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    '''아들'''


class sister(father):  # 자식클래스(부모클래스) 아빠매소드를 상속받겠다
    def pretty(self):
        print("예쁘다")

    def handsome(self):
        super().handsome()


brother = brother()
brother.handsome()

girl = sister()
girl.handsome()
girl.pretty()

#### 응용

class mother():
    def __init__(self, who):
        self.who = who

    def pretty(self):
        print("{}를 닮아 예쁘다".format(self.who))


class daughter(mother):
    def __init__(self, who, where):
        super().__init__(who)
        self.where = where

    def part(self):
        print("{} 말이야".format(self.where))

girl = daughter('엄마', '얼굴')
girl.pretty()
girl.part()


class mother():
    def __init__(self, who):
        self.who = who

    def pretty(self):
        print("{}를 닮아 예쁘다".format(self.who))


class daughter(mother):
    def __init__(self, who, where):
        super().__init__(who)
        self.where = where

    def part(self):
        print("{} 말이야".format(self.where))

    def pretty(self):
        super().pretty()
        self.part()

girl = daughter('엄마', '얼굴')
girl.pretty()

## nn.Parameter(텐서 객체, requires_grad=True)

#### 텐서 객체가 module의 attribute를 사용하기 위해서 이용
#### requires_grad True가 디폴트로 변화도 추적 가능

param1 = nn.Parameter(torch.ones(5))
param1

param2 = nn.Parameter(torch.zeros(5))
param2

## tensor.mean(input, dim, keepdim=False)
#### row 값들의 평균을 계산, input은 자기 자신
x = torch.rand(2, 5)
x
x.size()

#### dim 은 평균 내릴 rank를 의미
x1 = x.mean(0)
x1
x1.size()
x2 = x.mean(1)
x2
x2.size()

#### keepdim 이 True이면 원래 차원 규격을 유지
x3 = x.mean(-1, keepdim=True)
x3
x3.size()
x = torch.rand(2, 3, 4)
x
x4 = x.mean(-1, keepdim=True)
x4
x4.size()

## torch.sqrt(input, out=None)
#### tensor 안의 각 요소들에 대해 루트를 적용한 텐서 객체 반환
x = torch.randn(2, 3)
x
torch.sqrt(x)

## nn.Embedding(총 단어의 갯수, 임베딩할 벡터 차원)