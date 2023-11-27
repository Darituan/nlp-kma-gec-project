from transformers import RobertaTokenizer
import tensorflow as tf


def get_target_vocab_size(target_path):

    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")

    with open(target_path, 'r', encoding='utf-8') as f:
        target_text = f.read()

    target_tokens = tokenizer.tokenize(target_text)

    unique_target_tokens = set(target_tokens)

    num_unique_target_tokens = len(unique_target_tokens)

    return num_unique_target_tokens


def get_target_max_seq_len(target_path):

    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")

    with open(target_path, 'r', encoding='utf-8') as f:
        target_data = f.readlines()

    max_target_length = 0
    for text in target_data:
        encoded_target = tokenizer.encode(text, add_special_tokens=True)
        max_target_length = max(max_target_length, len(encoded_target))

    return max_target_length


def construct_dataset(input_path, target_path, max_length, batch_size):

    with open(input_path, 'r', encoding='utf-8') as f:
        incorrect_sentences = f.readlines()

    with open(target_path, 'r', encoding='utf-8') as f:
        corrected_sentences = f.readlines()

    sentence_pairs = list(zip(incorrect_sentences, corrected_sentences))

    tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")

    tokenized_sentences = tokenizer(incorrect_sentences, text_pair=corrected_sentences, padding=True, truncation=True,
                                    max_length=max_length, return_tensors='tf')

    input_ids = tokenized_sentences['input_ids']

    target_ids = tf.concat([input_ids[:, 1:], tf.fill((input_ids.shape[0], 1), tokenizer.pad_token_id)], axis=1)

    dataset = tf.data.Dataset.from_tensor_slices((input_ids, target_ids)).batch(batch_size)

    return dataset


def main():

    input_path = '../data/gec-only/train.src.tok'
    target_path = '../data/gec-only/train.tgt.tok'

    print(get_target_max_seq_len(input_path))


if __name__ == '__main__':
    main()
