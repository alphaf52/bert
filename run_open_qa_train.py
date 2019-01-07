
import collections
import json
import math
import os
import random
import modeling
import optimization
import tokenization
import six
import tensorflow as tf
import numpy as np

NUM_DOCS = 2
NUM_ANSWER_SPANS = 20

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "train_file", None,
    "json file path for training. E.g., train_output.json")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_integer("train_batch_size", 3, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 2,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 30.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "verbose_logging", False,
    "If true, all of the warnings related to data processing will be printed. "
    "A number of warnings are expected for a normal SQuAD evaluation.")


class OpenQAExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self,
               qid,
               question_text,
               doc_tokens,
               orig_answer_text=None,
               start_positions=None,
               end_positions=None):
    self.qid = qid
    self.question_text = question_text
    self.doc_tokens = doc_tokens
    self.orig_answer_text = orig_answer_text
    self.start_positions = start_positions
    self.end_positions = end_positions

  def __str__(self):
    return self.__repr__()

  def __repr__(self):
    s = ""
    s += "id: %s" % (self.qid)
    s += ", question_text: %s" % (
        tokenization.printable_text(self.question_text))
    s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
    if self.start_position:
      s += ", start_positions: %s" % (self.start_position)
    if self.start_position:
      s += ", end_positions: %s" % (self.end_position)
    return s


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               tokens_list,
               input_ids_list,
               input_mask_list,
               segment_ids_list,
               start_positions=None,
               end_positions=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.tokens_list = tokens_list
    self.input_ids_list = input_ids_list
    self.input_mask_list = input_mask_list
    self.segment_ids_list = segment_ids_list
    self.start_positions = start_positions
    self.end_positions = end_positions

def read_open_qa_examples(inputfile, is_training):
  """Read a json file from DocumentQA into a list of OpenQAExample."""
  examples = []
  with open(inputfile, "r") as fin:
    for line in fin:
      item = json.loads(line.strip())
      qid = item["question_id"]
      question_text = " ".join(item["question"])
      doc_tokens = item["context"]
      orig_answer_text = item["answer_text"]
      start_positions = [[answer_span[0] for answer_span in x] for x in item["answer_spans"]]
      end_positions = [[answer_span[1] for answer_span in x] for x in item["answer_spans"]]
      example = OpenQAExample(
          qid, question_text, doc_tokens, orig_answer_text, start_positions, end_positions)
      examples.append(example)
  return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 max_query_length, is_training, output_fn):
  """Loads a data file into a list of `InputBatch`s."""

  unique_id = 1000000000
  c1, c2 = 0, 0 

  for (example_index, example) in enumerate(examples):
    query_tokens = tokenizer.tokenize(example.question_text)
    if len(query_tokens) > max_query_length:
      query_tokens = query_tokens[0:max_query_length]
      c1 += 1

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    if example_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("unique_id: %s" % (unique_id))
      if is_training:
        tf.logging.info("answer: %s" % (example.orig_answer_text))
    elif example_index % 100 == 0:
      tf.logging.info("example_index: %s" % (example_index))

    tokens_list = []
    input_ids_list = []
    segment_ids_list = []
    input_mask_list = []
    start_positions = []
    end_positions = []
    for i in range(NUM_DOCS):
      tok_to_orig_index = []
      orig_to_tok_index = []
      all_doc_tokens = []
      if i < len(example.doc_tokens):
        for (j, token) in enumerate(example.doc_tokens[i]):
          orig_to_tok_index.append(len(all_doc_tokens))
          """
          # TODO:
          token = token.replace("QQQQQQ", "%Q")
          token = token.replace("%%DOCUMENT%%", "%D")
          token = token.replace("%%PARAGRAPH%%", "%P")
          token = token.replace("%%PARAGRAPH_GROUP%%", "%G")
          sub_tokens = tokenizer.tokenize(token)
          """
          if not token in set(["%%DOCUMENT%%", "%%PARAGRAPH%%", "%%PARAGRAPH_GROUP%%"]):
            sub_tokens = tokenizer.tokenize(token)
          else:
            sub_tokens = []
          for sub_token in sub_tokens:
            tok_to_orig_index.append(j)
            all_doc_tokens.append(sub_token)

      tokens = []
      segment_ids = []
      tokens.append("[CLS]")
      segment_ids.append(0)
      for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
      tokens.append("[SEP]")
      segment_ids.append(0)
      for token in all_doc_tokens:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(tokens)
      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      input_mask = [1] * len(input_ids)

      # Truncate over long sequence
      if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]
        input_mask = input_mask[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        c2 += 1

      start_positions.append([])
      end_positions.append([])
      if i < len(example.doc_tokens):
        for j in range(len(example.start_positions[i])):
          sp = example.start_positions[i][j]
          sp = len(query_tokens) + 2 + orig_to_tok_index[sp]
          ep = example.end_positions[i][j]
          if ep != len(orig_to_tok_index) - 1:
            ep = len(query_tokens) + 2 + orig_to_tok_index[ep + 1] - 1
          else:
            ep = len(all_doc_tokens) - 1
          if sp < len(input_ids) and ep < len(input_ids):
            start_positions[-1].append(sp)
            end_positions[-1].append(ep)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      if example_index < 20:
        tf.logging.info("#%d" % i)
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info(
            "input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info(
            "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info(
            "start_positions: %s" % start_positions[-1])
        tf.logging.info(
            "end_positions: %s" % end_positions[-1])

      tokens_list.extend(tokens)
      input_ids_list.extend(input_ids)
      segment_ids_list.extend(segment_ids)
      input_mask_list.extend(input_mask)

    if all([len(sp) == 0 for sp in start_positions]):
        continue

    feature = InputFeatures(
        unique_id=unique_id,
        example_index=example_index,
        tokens_list=tokens_list,
        input_ids_list=input_ids_list,
        input_mask_list=input_mask_list,
        segment_ids_list=segment_ids_list,
        start_positions=start_positions,
        end_positions=end_positions)

    # Run callback
    output_fn(feature)

    unique_id += 1

  tf.logging.info("Num of overlong querys: %d" % c1)
  tf.logging.info("Num of overlong documents : %d" % c2)


def create_model(bert_config, is_training, input_ids_list, input_mask_list,
                 segment_ids_list, use_one_hot_embeddings):
  """Creates a classification model."""
  all_logits = []
  input_ids_shape = modeling.get_shape_list(input_ids_list, expected_rank=2)
  batch_size = input_ids_shape[0]
  seq_length = input_ids_shape[1]
  seq_length = seq_length // NUM_DOCS

  def reshape_and_unstack_inputs(inputs, batch_size):
      inputs = tf.reshape(inputs, [batch_size, NUM_DOCS, seq_length])
      return tf.unstack(inputs, axis=1)

  input_ids_list = reshape_and_unstack_inputs(input_ids_list, batch_size)
  input_mask_list = reshape_and_unstack_inputs(input_mask_list, batch_size)
  segment_ids_list = reshape_and_unstack_inputs(segment_ids_list, batch_size)

  start_logits, end_logits = [], []
  with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
    for i in range(len(input_ids_list)):
      model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=input_ids_list[i],
          input_mask=input_mask_list[i],
          token_type_ids=segment_ids_list[i],
          use_one_hot_embeddings=use_one_hot_embeddings,
          scope="bert")
      final_hidden = model.get_sequence_output()

      final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
      hidden_size = final_hidden_shape[2]

      output_weights = tf.get_variable(
          "cls/open_qa/output_weights", [2, hidden_size],
          initializer=tf.truncated_normal_initializer(stddev=0.02))
      output_bias = tf.get_variable(
          "cls/open_qa/output_bias", [2], initializer=tf.zeros_initializer())

      final_hidden_matrix = tf.reshape(final_hidden,
                                       [batch_size * seq_length, hidden_size])
      logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)

      logits = tf.reshape(logits, [batch_size, seq_length, 2])
      logits = tf.transpose(logits, [2, 0, 1])
      unstacked_logits = tf.unstack(logits, axis=0)
      (s_logits, e_logits) = (unstacked_logits[0], unstacked_logits[1])
      start_logits.append(s_logits)
      end_logits.append(e_logits)

  start_logits = tf.concat(start_logits, axis=-1)
  end_logits = tf.concat(end_logits, axis=-1)

  return (start_logits, end_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids_list = features["input_ids_list"]
    input_mask_list = features["input_mask_list"]
    segment_ids_list = features["segment_ids_list"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids_list=input_ids_list,
        input_mask_list=input_mask_list,
        segment_ids_list=segment_ids_list,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids_list)[1]

      def compute_loss(logits, positions, weights):
        a = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        b = tf.expand_dims(weights, -1)
        c = tf.multiply(a, b)
        d = tf.reduce_sum(c, 1) / tf.expand_dims(tf.reduce_sum(weights, -1), -1) # TODO:
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(d * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      weights = tf.cast(features["weights"], tf.float32)

      start_loss = compute_loss(start_logits, start_positions, weights)
      end_loss = compute_loss(end_logits, end_positions, weights)
      total_loss = (start_loss + end_loss) / 2.0

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError(
          "Only TRAIN and PREDICT modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids_list": tf.FixedLenFeature([NUM_DOCS * seq_length], tf.int64),
      "input_mask_list": tf.FixedLenFeature([NUM_DOCS * seq_length], tf.int64),
      "segment_ids_list": tf.FixedLenFeature([NUM_DOCS * seq_length], tf.int64),
  }

  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([NUM_ANSWER_SPANS], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([NUM_ANSWER_SPANS], tf.int64)
    name_to_features["weights"] = tf.FixedLenFeature([NUM_ANSWER_SPANS], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training, max_seq_length):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)
    self.max_seq_length = max_seq_length

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids_list"] = create_int_feature(feature.input_ids_list)
    features["input_mask_list"] = create_int_feature(feature.input_mask_list)
    features["segment_ids_list"] = create_int_feature(feature.segment_ids_list)

    if self.is_training:
      start_positions = []
      for i in range(len(feature.start_positions)):
        for sp in feature.start_positions[i]:
          start_positions.append(i * self.max_seq_length + sp)
      end_positions = []
      for i in range(len(feature.end_positions)):
        for ep in feature.end_positions[i]:
          end_positions.append(i * self.max_seq_length + ep)
      weights = [1] * len(start_positions) + [0] * (NUM_ANSWER_SPANS - len(start_positions))
      start_positions = start_positions + [0] * (NUM_ANSWER_SPANS - len(start_positions))
      end_positions = end_positions + [0] * (NUM_ANSWER_SPANS - len(end_positions))
      features["start_positions"] = create_int_feature(start_positions[:NUM_ANSWER_SPANS])
      features["end_positions"] = create_int_feature(end_positions[:NUM_ANSWER_SPANS])
      features["weights"] = create_int_feature(weights[:NUM_ANSWER_SPANS])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.train_file:
    raise ValueError(
        "`train_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  validate_flags_or_throw(bert_config)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None

  train_examples = read_open_qa_examples(
      inputfile=FLAGS.train_file, is_training=True)
  num_train_steps = int(
      len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  # Pre-shuffle the input to avoid having to make a very large shuffle
  # buffer in in the `input_fn`.
  rng = random.Random(12345)
  rng.shuffle(train_examples)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  filename = os.path.join(FLAGS.output_dir, "train.tf_record")
  if True: # not os.path.exists(filename):
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
        filename=filename,
        is_training=True,
        max_seq_length=FLAGS.max_seq_length)
    convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=FLAGS.max_seq_length,
        max_query_length=FLAGS.max_query_length,
        is_training=True,
        output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", len(train_examples))
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    del train_examples

  train_input_fn = input_fn_builder(
      input_file=filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
  estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
