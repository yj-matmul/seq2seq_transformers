# modeling.py : 메인 BERT 모델과 관련 함수들...

"""
  Author : jaehyung.vincent.ko
  < 아래 한글 멘트들에 대한 shape 관련해서 >
  0. 모든 멘트는 base model 기준이다.
    0.1 base model : L-12_H-768_A-12 (레이어 12개, 히든 사이즈는 768, 어텐션 헤드는 12)
    0.2 large model : L-24_H-1024_A-16 (레이어 24개, 히든 사이즈는 1024, 어텐션 헤드는 16)
  1. batch_size = 32
    1.1 BERT README.md 파일에 'Sentence (and sentence-pair) classification tasks' 기준으로
    1.2 'train_batch_size=32' 로 되어 있으므로, 32로 정의한다.
  2. seq_length = 512
    2.1 max_position_embeddings보다 작거나 같아야 하는데
    2.2 max_position_embeddings = 512 이므로
    2.3 seq_length = 512로 정의한다.
  3. embedding_size (= hidden_size) = 768
    3.1 embedding_lookup 함수에는 128로 되어 있고,
    3.2 다른 코드에는 'embedding_size=config.hidden_size'로 되어 있다.
    3.3 따라서, 아래 멘트들에서는 embedding_size를 hidden_size에 맞춰서 768로 넣고 정리한다.
    3.4 embedding_size = 768
  4. token_type_vocab_size = 2
    4.1 원본인 tf 코드에는 16으로 되어 있다.
    4.2 huggingface의 pytorch 코드에는 2로 되어 있다.
    4.3 NSP(Next Sentence Prediction)에서는 다음 문장 예측이라, sentence가 2개이므로, type이 2개이면 될 것 같은데, 왜 16인가?
    4.4 조사가 필요하지만, 실제 실행시에는 token_type_vocab_size = 2로 실행해야 할 것으로 보인다.
  5. num_attention_heads = 12
    5.1 attention_layer 함수에서 이 값이 1로 되어 있는데, BertConfig를 따라야 한다.
    5.2 따라서, 12로 정의한다.
  6. size_per_head = 64
    6.1 이 값은 'hidden_size / num_attention_heads' 이므로, 64로 정의한다. (= 768 / 12)
    6.2 attention_layer 함수에서 이 값이 512로 되어 있는데, huggingface 코드를 보면, 'attention_head_size = 64' 이다.
"""

from __future__ import absolute_import    # 임포트: 복수 줄 및 절대/상대
from __future__ import division           # 나누기 연산자 변경
from __future__ import print_function     # print를 함수로 만들기

import collections          # 컨테이너
import copy
import json
import math
import re                   # regular expression
import numpy as np
import six                  # Python 2와 3 호환성 라이브러리. Python 버전 간의 차이점을 완화하고 Python 버전 모두에서 호환되는 Python 코드를 작성하기위한 유틸리티 함수를 제공
import tensorflow as tf

# BERT Model에 대한 Configuration 정의, 유틸리팀 함수 포함
class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: `inputs_ids`에 대한 Vocabulary 사이즈

      hidden_size: encoder 레이어 와 the pooler 레이어 사이즈 ==> 768

      num_hidden_layers: Transformer encoder 내의 hidden 레이어 수 ==> 12

      num_attention_heads: Transformer encoder의 각 attention layer에 대한 attention head 수 ==> 12

      intermediate_size: Transformer encoder의 "intermediate" 레이어의 사이즈 (i.e., feed-forward) ==> 3072

      hidden_act: encoder 와 pooler 에서의 non-linear activation function (function or string) ==> gelu

      hidden_dropout_prob: embeddings, encoder, and pooler 내의 모든 fully connected layer 에서의 dropout probability ==> 0.1

      attention_probs_dropout_prob: attention probabilities 에 대한 dropout ratio ==> 0.1

      max_position_embeddings: 모델이 사용하는 최대 시퀀스 길이 Typically set this to something large just in case (e.g., 512 or 1024 or 2048). ==> 512

      type_vocab_size: BERT 모델에서 사용하는 `token_type_ids`에 대한 Vocabulary 사이즈 ==> 16

      initializer_range: 모든 weight 매트릭스를 초기화를 위한 truncated_normal_initializer의 표준편차 ==> 0.02
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  """ 파이썬 딕셔너리 파라미터로부터 BertConfig를 만든다. """
  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  """ json 파일을 읽어와서 파라미터로부터 BertConfig를 만든다. """
  """ json.loads ==> json을 dictionary로 변환 """
  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output
  
  """ json.dumps ==> dictionary를 json으로 변환 """
  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


# BERT 모델에 전체적인 부분. 가장 핵심임.
class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. training이면 true, evaluation이면 false, dropout 적용 여부를 컨트롤 한다.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    # config 정보를 deepcopy한다. 즉, 새로운 객체로 완전 복사
    config = copy.deepcopy(config)
    # eval mode이면, dropout은 적용하지 않는다. dropout은 당연히 training에만 적용
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0] # 32
    seq_length = input_shape[1] # 512

    # 인풋 마스크가 none이면, input_ids 사이즈와 동일하면서 값이 1로 구성된 텐서 생성
    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)  # [32, 512]

    # token_type_ids가 none이면, input_ids 사이즈와 동일하면서 값이 0으로 구성된 텐서 생성
    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    # 실행 순서 : with tf.variable_scope("embeddings") ==> with tf.variable_scope("encoder") ==> with tf.variable_scope("pooler")
    with tf.variable_scope(scope, default_name="bert"):
      # 임베딩을 수행하는 블럭
      with tf.variable_scope("embeddings"):
        # word ids에 대한 임베딩 룩업을 수행한다.
        # embedding_lookup 함수를 통해 임베딩 된 벡터 객체와 weight matrix(embedding_table)를 받는다.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # 포지션 임베딩과 토큰 타입 임베딩을 더하고, 레이어 노멀라이제이션과 드랍아웃을 수행한 텐서를 아웃풋으로 받는다.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      # 인코딩을 수행하는 블럭, 트랜스포머
      with tf.variable_scope("encoder"):
        # 사이즈가 [batch_size, seq_length]인 2d 마스크를 사이즈가 [batch_size, seq_length, seq_length]인 3d 마스크로 만든다.
        # 해당 마스크는 어텐션 스코어를 계산할 때 사용된다.
        # [32, 512] ==> [32, 512, 512]
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # stacked 트랜스포머 실행
        # `sequence_output` shape = [batch_size, seq_length, hidden_size] = [32, 512, 768]
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size, # 768
            num_hidden_layers=config.num_hidden_layers, # 12
            num_attention_heads=config.num_attention_heads, # 12
            intermediate_size=config.intermediate_size, # 3072
            intermediate_act_fn=get_activation(config.hidden_act), # 실행할 activation function 정하기
            hidden_dropout_prob=config.hidden_dropout_prob, # 0.1
            attention_probs_dropout_prob=config.attention_probs_dropout_prob, # 0.1
            initializer_range=config.initializer_range, # 0.2
            do_return_all_layers=True)

      # 인코더 마지막 레이어를 아웃풋으로 받는다.
      # 아웃풋은 ==> float Tensor of shape [batch_size, seq_length, hidden_size]
      self.sequence_output = self.all_encoder_layers[-1]
      
      # "pooler" 역할 : encoded sequence tensor의 사이즈 전환
      #  ==> [batch_size, seq_length, hidden_size] ==> [batch_size, hidden_size]
      # segment-level(or segment-pair-level) 분류 타스크에서 반드시 필요하기 때문. : cls 토큰만 남겨서 분류 타스크에 사용하기 위해서
      # 즉, segment에 대한 고정된 dimensional representation을 필요로 하는 타스크들이기 때문
      with tf.variable_scope("pooler"):
        # 단순히 첫번째 토큰에 해당하는 hidden state를 취한다.(pool한다.)
        # We assume that this has been pre-trained (프리트레이닝 되었다고 전제한다.)
        
        # tf.squeeze 함수를 사용해 시퀀스의 첫번째 토큰으로만 구성된 텐서를 만든다.
        # [batch_size, seq_length, hidden_size] ==> [batch_size, 1, hidden_size] ==> [batch_size, hidden_size]
        # [batch_size, hidden_size] : 여기에는 시퀀스의 첫번째 토큰 정보만 남아 있다. 즉, cls 토큰만 남는다.
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        
        # 시퀀스의 첫번째 토큰으로만 구성된 텐서를 input으로 하는 FC(Fully-Connected) Layer 구성, 해당 레이어는 768 hidden units
        # 여기서 만들어진 hidden layer에 cls 토큰 정보가 pre-training 하고 난 cls 토큰에 다시 한번 더 FC 연산을 하여 만들어 두는 것이다.
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  """
    아래 def로 정의된 함수를 호출하는 순서는...
      1. get_embedding_table
      2. get_embedding_output
      3. get_all_encoder_layers
      4. get_sequence_output
      5. get_pooled_output
  """
  
  # pooler를 호출하는 함수
  def get_pooled_output(self):
    return self.pooled_output

  # 트랜스포머 인코더의 마지막 히든 레이어를 아웃풋으로 얻는 함수
  # float 텐서를 받으며, 사이즈는 [batch_size, seq_length, hidden_size] 즉, [32, 512, 768]이다.
  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  # 트랜스포머 인코더 작업을 호출하는 함수
  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  # 트랜스포머 인풋으로 넣기 위한 텐서를 임베딩을 통해 아웃풋으로 받는 함수
  # float 텐서를 받으며, 사이즈는 [batch_size, seq_length, hidden_size] 즉, [32, 512, 768]이다.
  # 그리고 token_embedding + position embedding + segment embedding ==> layer norm ==> dropout 까지 연산을 거친 텐서이다.
  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


# Activation Function ==> GELUs (Gaussian Error Linear Units)
def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


# activation_string을 인자로 받아서, 소문자로 치환하고 실행할 activation function을 리턴해 준다.
# linear : none , relu : tf.nn.relu , gelu : gelu , tanh : tf.tanh
def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


# 현재 변수와 체크포인트 변수를 합산해서 리턴해 준다.
def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


# dropout을 수행하는 함수
def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output


# input tensor의 마지막 dimension에 대해 layer normalization을 수행하는 함수
def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


# layer normalization을 수행한 이후 dropout도 수행하는 함수
def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


# 초기화 함수 : 'truncated_normal_initializer' 사용, 표준편차(std_dev)는 0.02 사용
def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


# id 텐서에 대한 word embedding - 벡터 값 lookup
# 아마도 vocab.txt 파일에 있는 토큰들은 unique할 것이다. ==> 추후 체크 요망
# 여기 embedding_lookup 로직을 보면, input_ids에 대해 임베딩을 진행한다. 즉, setence 데이터 수와 그 토큰 들이다.
# ==> 이러면, 중복 값 토큰들이 존재한다.
# question ==> 이미 여기서 동일 토큰에 대한 다른 의미의 임베딩 벡터들이 만들어지는 것은 아닌가? ==> 추적 조사가 필요하다.
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: word ids(인덱스?)를 포함하는 [batch_size, seq_length] shape을 가지는 텐서, dtype = int32
    vocab_size: int. 임베딩 하려는 vocab 사이즈
        'uncased_L-12_H-768_A-12' ==> 30,522
        'uncased_L-24_H-1024_A-16' ==> 30,522
        'cased_L-12_H-768_A-12' ==> 28,996
        'cased_L-24_H-1024_A-16' ==> 28,996
        'multilingual_L-12_H-768_A-12' ==> 105,879
        'multi_cased_L-12_H-768_A-12' ==> 119,547
    embedding_size: word 임베딩의 Width. 즉, word 벡터의 임베딩 후의 벡터 사이즈, dtype = int
    initializer_range: 임베딩 초기화 range. dtype = float
    word_embedding_name: 임베딩 테이블 이름. dtype = string
    use_one_hot_embeddings: bool. If True, use one-hot method for word embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  
  # 현재 함수는 input shape을 [batch_size, seq_length, num_inputs]로 가정한다.
  # 만약, input이 2d 텐서이면, 즉, shape이 [batch_size, seq_length] 이면, [batch_size, seq_length, 1]로 reshape 한다.
  # question : num_inputs 를 왜 넣어줄까? 아마도 num_inputs은 input 데이터 전체 갯수인것 같은데, why???

  # input_ids의 shape이 2 dimension이면 맨 끝에 임의 dimension을 추가하여 3d 텐서로 만든다.
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  # lookup 하기 위한 임베딩 테이블 설정, 2차원이며, row는 vocab 수, column은 임베딩 후의 각 토큰의 벡터 사이즈
  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  # input_ids를 1차원 텐서로 reshape한 flat한 텐서를 만든다.
  flat_input_ids = tf.reshape(input_ids, [-1])

  # use_one_hot_embeddings이 true이면, one hot으로 만든다. ==> [batch_size * seq_length, vocab_size]
  # use_one_hot_embeddings이 false이면, 임베딩 테이블에서 input_ids를 1차원 텥서로 만든 flat_input_ids 위치의 vector 값을 lookup 한다.
  # 즉, use_one_hot_embeddings이 false이면, one hot 인코딩을 사용하는 것이 아니고, keras 임베딩에서 사용하는 integer 인코딩을 사용하는 것으로 보인다.
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    # one hot 벡터와 임베딩 테이블의 matrix multiplication
    # [batch_size * seq_length, vocab_size] matmul [vocab_size, embedding_size(768)] = [batch_size * seq_length, embedding_size(768)]
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)

  # return되는 output 텐서를 reshape한다.
  # ==> 3차원 input_ids의 앞의 2차원[batch_size, seq_length]는 그대로 두고, 마지막 차원(3차원)과 임베딩 사이즈를 곱하여 최종 임베딩이 완료된 텐서를 리턴한다.
  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


# 임베딩 후속프로세스 처리, 'segment embedding + position embedding' 처리
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: embedding_lookup 함수를 통해 나온 텐서. shape ==> [batch_size, seq_length, embedding_size], dtype = float
    use_token_type: bool. Whether to add embeddings for `token_type_ids`. : segment embedding 여부
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. token type ids에 대한 임베딩 테이블 변수 이름
    use_position_embeddings: bool. 시퀀스 각 토큰 포지션에 대한 position embedding 여부
    position_embedding_name: string. positional embedding에 대한 임베딩 테이블 변수 이름
    initializer_range: float. weight 초기화 범위
    max_position_embeddings: int. 최대 시퀀스 길이. 인풋 텐서의 시퀀스 길이보다 더 길수도 있으나, 더 짧을 수는 없다.
    dropout_prob: float. 마지막 아웃풋 텐서에 적용되는 dropout 확률

  Returns:
    인풋 텐서와 같은 shape을 가지는 float 텐서

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3) # [batch_size, seq_length, embedding_size]
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor # 인풋 텐서 따로 저장해 놓고,

  if use_token_type: # segment embedding을 사용한다면,
    if token_type_ids is None: # [batch_size, seq_length] 크기인 token_type_ids가 none이면, 에러 발생시킨다.
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    # 토큰_타입 임베딩 테이블 정의
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width], # [16, 768]
        initializer=create_initializer(initializer_range))
    
    # 'token_type_vocab_size'가 16이다. pytorch 코드는 2로 되어 있다. question ==> 2가 맞는 것 아닌가? 여기는 왜 16으로 되어 있을까?
    # 여기는 vocab 사이즈가 16으로 작기 대문에 one-hot을 사용한다. 작은 vocab에서는 항상 더 빠르기 때문에.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1]) # 1d 텐서로 reshape [batch_size * seq_length]
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size) # 16 vocab size로 one-hot ==> [batch_size * seq_length, 16]
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table) # [batch_size * seq_length, 16] matmul [16, 768] = [batch_size * seq_length, 768]
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width]) # 3d 텐서로 reshape ==> [batch_size, seq_length(512), width(768)]
    output += token_type_embeddings # 처음 인풋 텐서에 segment embedding 값을 더한다.

  if use_position_embeddings: # position embedding을 사용하다면,
    # seq_length 값이 max_position_embeddings(512) 값보다 작거나 같으면 ok, max_position_embeddings(512) 값보다 크다면 에러를 발생시킨다.
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width], # [512, 768]
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      
      # 만약, sentence의 sequence length가 max_position_embeddings(512)보다 작다면, position embedding에 대해서 slicing을 해서 크기를 맞춰 준다.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list()) # segment embedding까지 (+)된 output 텐서의 dimension 수 저장

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.

      # 아래 코드는 position_embeddings을 batch_size만큼 boradcasting을 하고 그 값을 이전 텐서에 더해 준다.
      # 여기서 이전 텐서는 'input_tensor + segment embedding' 까지 적용된 텐서이다.
      # 아래 코드에 대한 정확한 이해는 나중에 다시... ==> 내가 파이썬이 약해서리...
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  # 'input_tensor + segment embedding + position embedding'까지 진행된 아웃풋에 layer normalization과 dropout을 적용한 후 최종 아웃풋 텐서를 리턴한다.
  output = layer_norm_and_dropout(output, dropout_prob)
  return output


# 진행중, 2d 텐서 마스크로부터 3d 어텐션 마스크를 만든다.
# from_tensor : input_ids ==> [32, 512, 1]
# to_mask : 값이 1인 2d 텐서 ==> [32, 512]
def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D 또는 3D 텐서 ==> shape [batch_size, from_seq_length, ...].
    to_mask: int32 텐서 ==> shape [batch_size, to_seq_length].

  Returns:
    shape이 [batch_size, from_seq_length, to_seq_length]인 float 텐서
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3]) # from 텐서 shape 알기
  batch_size = from_shape[0]      # 32
  from_seq_length = from_shape[1] # 512

  to_shape = get_shape_list(to_mask, expected_rank=2) # to_mask shape 알기
  to_seq_length = to_shape[1]     # 512 , 값이 모두 1

  # to_mask 텐서의 shape을 2d에서 3d로 변경 : [32, 512] ==> [32, 1, 512] , 값이 모두 1
  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1] = [32, 512, 1]
  # `broadcast_ones`는 from 텐서의 값을 가지고 있다. 임베딩 텐서가 아닌, 원래 인풋 데이터에 대한 텐서이다.
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  # mask ==> [batch_size, from_seq_length, 1] * [batch_size, 1, to_seq_length] = [batch_size, from_seq_length, to_seq_length]
  # from 텐서와 전부 값이 1인 to_mask 텐서를 곱한다. 따라서, from 텐서에서 값이 0이면 0 값이 되고, 값이 0이 아니면 원래 값을 그대로 유지한다.
  # [32, 512, 1] * [32, 1, 512] = [32, 512, 512]
  mask = broadcast_ones * to_mask

  return mask


# 어텐션
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """from 텐서에서 to 텐서에 multi-headed attention을 수행한다.

  BERT에서는 트랜스포머 인코더만 사용하므로, 즉, self attention만 사용하므로, from 텐서와 to 텐서는 동일한 텐서이다.
  각 time step에서 from 텐서가 to 텐서를 어텐션 한다. 그리고 고정 길이 벡터를 리턴한다.

  먼저, from 텐서를 가지고 query 텐서로 projection 한다.
  그리고, to 텐서를 가지고 key 텐서, value 텐서로 projection 한다.
  이러한 query, key, value 텐서는 'num_attention_heads'(=12) 갯수 만큼의 탠서 List 들이다.
  텐서 List의 각각은 shape이 [batch_size, seq_length, size_per_head] = [32, 512, 64] 이다.

  *** projecttion : 별 거 없다. shape에 맞게 weight matrix를 정의하고 dot product 하면 나오는 거다. ***

  이제 query 텐서와 key 텐서를 dot product 하고 scaling 한다.
  그리고, 어텐션 확률 값을 얻기 위해 softmax 함수를 취하고,
  이렇게 계산된 어텐션 확률 값을 가지고 value 텐서 에다가 interpolation 한다. 즉, 곱한다.

  num_attention_heads(= 12) 였으므로, 이렇게 나온 텐서들이 12개가 된다.
  각 텐서의 크기가 [32, 512, 64] 라는 것을 반드시 기억하자.

  마지막으로, 이러한 12개 텐서들을 하나의 텐서로 concatenation 한다.
  그러면, 다시 64 * 12 이므로, 768 사이즈가 된다.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads, # input_tensor ==> [32 * 512, 768]
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width]) # [32, 512, 12, 64]

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3]) # [32, 12, 512, 64]
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  # from 텐서와 to 텐서의 shape 크기가 맞지 않으면 에러
  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]        # 32
    from_seq_length = from_shape[1]   # 512
    to_seq_length = to_shape[1]       # 512
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)  = 32
  #   F = `from_tensor` sequence length     = 512
  #   T = `to_tensor` sequence length       = 512
  #   N = `num_attention_heads`             = 12
  #   H = `size_per_head`                   = 64

  from_tensor_2d = reshape_to_matrix(from_tensor) # [32 * 512, 768]
  to_tensor_2d = reshape_to_matrix(to_tensor)     # [32 * 512, 768]

  # `query_layer` = [B*F, N*H] = [32 * 512, 12 * 64]
  query_layer = tf.layers.dense(            # [32 * 512, 768] matmul [12 * 64] = [32 * 512, 768]
      from_tensor_2d,                       # [32 * 512, 768]
      num_attention_heads * size_per_head,  # [12 * 64]
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H] = [32 * 512, 12 * 64]
  key_layer = tf.layers.dense(              # [32 * 512, 768] matmul [12 * 64] = [32 * 512, 768]
      to_tensor_2d,                         # [32 * 512, 768]
      num_attention_heads * size_per_head,  # [12 * 64]
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H] = [32 * 512, 12 * 64]
  value_layer = tf.layers.dense(            # [32 * 512, 768] matmul [12 * 64] = [32 * 512, 768]
      to_tensor_2d,                         # [32 * 512, 768]
      num_attention_heads * size_per_head,  # [12 * 64]
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  # [32 * 512, 768] shape이 들어가서 아웃풋 텐서는 [32, 12, 512, 64]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  # [32 * 512, 768] shape이 들어가서 아웃풋 텐서는 [32, 12, 512, 64]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]

  # [32, 12, 512, 64] matmul [32, 12, 64, 512] = [32, 12, 512, 512]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  # size_per_head의 루트 값을 가지고 scaling
  # 어텐션 스코어 * (1 / 루트(64)) = 어텐션 스코어 * (1 / 8)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:    # 현재 [32, 512, 512]
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1]) # [32, 512, 512] ==> [32, 1, 512, 512]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  # [32 * 512, 768] ==> [32, 512, 12, 64]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  # [32, 512, 12, 64] ==> [32, 12, 512, 64]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  # [32, 12, 512, 512] matmul [32, 12, 512, 64] = [32, 12, 512, 64]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  # [32, 12, 512, 64] ==> [32, 512, 12, 64]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    # [32, 512, 12, 64] ==> [32 * 512, 12 * 64]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    # [32, 512, 12, 64] ==> [32, 512, 12 * 64]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


# 트랜스포머 인코더
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  # 768 / 12 의 나머지가 0이 아니면 에러
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  # attention_head_size = 768 / 12 = 64
  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]   # 32
  seq_length = input_shape[1]   # 512
  input_width = input_shape[2]  # 768

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  # 트랜스포머는 모든 레이어에서 sum residual을 수행하므로, 인풋과 히든 레이어는 사이즈가 동일해야 한다.
  # 3차원 인풋의 마지막 사이즈(768)가 히든 레이어 사이즈(768)와 같지 않으면 에러
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  # 2d, 3d reshapeing을 피하기 위해 3d 텐서를 2d 텐서로 reshape
  # [batch_size, seq_length, embedding_size] ==> [batch_size * seq_length, embedding_size] = [32 * 512, 768]
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  # 레이어 12개를 거친다. for문이 12번 돈다.
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output # 2차원으로 만든 텐서를 레이어 인풋으로 넣는다.

      # 1. attention 처리를 하고...
      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,  # self attention 이므로 from 텐서와 to 텐서는 동일한 텐서 [32, 512, 768]
              to_tensor=layer_input,    # self attention 이므로 from 텐서와 to 텐서는 동일한 텐서 [32, 512, 768]
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # 2. 히든 유닛 갯수가 3072인 중간 레이어 하나 거치고...
      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # 3. 최종으로 다시 히든 유닛 갯수가 768인 아웃풋 레이어를 만든다.
      # 그래야 다름 for문에서 그 다음 레이어의 인풋으로 넣을 수 있다. 즉, 사이즈를 맞춰야 하므로...
      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      # layer_output :  2d , input_shape : 3d의 원래 shape ==> 2개 인자 넣어서 original 3d shape으로 리턴 받는다.
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    # layer_output :  2d , input_shape : 3d의 원래 shape ==> 2개 인자 넣어서 original 3d shape으로 리턴 받는다.
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


# tensor의 shape list 정보를 알려주는 함수
# static dimension : return integer , dynamic dimension : tf. Tensor scalar
def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: shape을 찾기를 원하는 tf tensor object
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    tensor의 shape dimension list
    모든 static dimension은 파이썬 integer로 리턴된다.
    dynamic dimension은 tf.Tensor 스칼라로 리턴될 것이다.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


# input tensor의 shape을 matrix로 reshape 하여 리턴해 주는 함수
# width = input_tensor.shape[-1] ==> 기존 shape의 마지막 dimension 숫자를 width로...
# tf.reshape(input_tensor, [-1, width]) ==> width를 제외한 나머지 dimension들을 다 곱하여 알아서 계산 및 정리하라는 뜻
def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


# matrix로부터 해당 tensor의 original shape으로 reshape해 주는 함수
# 어떻게 맞게 변환이 되는지 계산이 완벽히 이해가 안됨. 나중에 다시 봐야 함.
def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


# 만약 tensor의 rank가 예상하고 있는 rank가 아니라면 exception error를 알려주는 함수
def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
