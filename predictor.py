import json
import re
import string
from functools import partial
from typing import List, Pattern, Optional

import torch
from pkg_resources import resource_filename
from tokenizers.implementations import BertWordPieceTokenizer
from torch import nn
from tqdm import tqdm

from configuration import Configuration
from data import preprocess_data, create_data_loader
from model import PuncRec

config = Configuration()

punctuation_enc = config.punctuation_encoding

inv_punctuation_enc = {v: k for k, v in punctuation_enc.items()}

TOKEN_RE = re.compile(r'-?\d*\.\d+|[a-zа-яё]+|-?\d+|\S', re.I)


def tokenize_text_simple_regex(txt: str, regex: Pattern, min_token_size: int = 0) -> List[str]:
    """Tokenize text with simple regex
    Args:
        txt: text to tokenize
        regex: re.compile output
        min_token_size: min char length to highlight as token
    Returns:
        tokens list
    """

    txt = txt.lower()
    all_tokens = regex.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize(corpus: List[str]) -> List[List[str]]:
    """Tokenize text corpus with simple regex
    Args:
        corpus: text corpus
    Returns:
        List of tokenized texts
    """
    tokenized_corpus = []
    for doc in corpus:
        tokenized_corpus.append(tokenize_text_simple_regex(doc, TOKEN_RE))

    return tokenized_corpus


def make_labeling(tokenized_corpus: List[List[str]], save_path: Optional[str] = None) -> List[List[str]]:
    """
    Make labeling to correspond BertPunc input data https://github.com/IsaacChanghau/neural_sequence_labeling/tree/master/data/raw/LREC
    Args:
        tokenized_corpus: tokenized text corpus
        save_path: path to save labeling result
    Returns:
        labeled tokenized text corpus
    """
    labeled_tokens = []
    for text_tokenized in tokenized_corpus:
        text_tokenized.append("")
        for i in range(len(text_tokenized) - 1):
            if text_tokenized[i] in string.punctuation:
                if text_tokenized[i + 1] == ".":
                    labeled_tokens[-1][1] = "PERIOD"
                elif text_tokenized[i + 1] == ",":
                    labeled_tokens[-1][1] = "COMMA"
                else:
                    continue
            else:
                if text_tokenized[i + 1] == ".":
                    labeled_tokens.append([text_tokenized[i], "PERIOD"])
                elif text_tokenized[i + 1] == ",":
                    labeled_tokens.append([text_tokenized[i], "COMMA"])
                else:
                    labeled_tokens.append([text_tokenized[i], "0"])
    if save_path is not None:
        with open(save_path, "w") as f:
            for token, label in labeled_tokens:
                f.write(f"{token}\t{label}\n")

    return labeled_tokens


def predictions(data_loader, bert_punc, device):
    y_pred = []
    y_true = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader), disable=True):
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            output = bert_punc(inputs)
            y_pred += list(output.softmax(dim=2).argmax(dim=2).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
    return y_pred, y_true


def right_decode_predictions(data_test, predictions, tokenizer, punctuation_enc, segment_size):
    """It takes linear time to execute
    """

    temp_X_test, _ = preprocess_data(data_test, tokenizer, punctuation_enc, segment_size)
    temp_X_test_decoded = []
    temp_X_test_encoded = []

    substitution = {0: "", 1: ',', 2: '.', 3: '?'}

    index_count = 3

    merged_encoded = []

    for i, (encoded_str, pred_y) in enumerate(zip(temp_X_test, predictions)):

        encoded_str = encoded_str[((segment_size - 1) // 2 - 1):]
        encoded_str = encoded_str[:segment_size // 2]
        encoded_str = encoded_str.tolist()
        _s = tokenizer.decode(encoded_str)

        _s = _s.replace(" [PAD]", substitution[pred_y])

        temp_X_test_decoded.append(_s)
        temp_X_test_encoded.append(encoded_str)

        if i == 0:
            merged_encoded += encoded_str
            continue

        merged_encoded.insert(index_count, 0)
        index_count += 2

        # TODO REVISE (problem with SEP and CLS token)
        #     if pred_y == 0:
        #         l1.insert(index_count, 0)
        #         index_count += 2
        #     else:
        #         enc_punct = tokenizer.encode(substitution[pred_y])
        #         l1[index_count: index_count + len(enc_punct) + 1] = enc_punct
        #         index_count += 2 + len(enc_punct)

        merged_encoded.append(encoded_str[-1])

    final_s = tokenizer.decode(merged_encoded)

    final_s_pad_tokenized = final_s.split(" [PAD]")

    filled_final_s = ""
    for i in range(len(predictions[:len(final_s_pad_tokenized)])):
        filled_final_s += final_s_pad_tokenized[i] + substitution[predictions[i]]

    filled_final_s = re.sub(r"< empty >\W", "", filled_final_s)
    return filled_final_s


def make_single_text_pred(domain_text, func_to_pred, segment_size, batch_size,
                          tokenizer, punctuation_enc, device):
    _prepared_domain_text = make_labeling(tokenize([domain_text]))
    prepared_domain_text = []
    for token, label in _prepared_domain_text:
        prepared_domain_text.append(f"{token}\t{label}\n")

    if len(prepared_domain_text) < segment_size:
        prepared_domain_text = prepared_domain_text[:] + ["<empty>\tO\n"] * (segment_size - len(prepared_domain_text))
        X_domain, y_domain = preprocess_data(prepared_domain_text, tokenizer, punctuation_enc, segment_size)
        data_loader_one_shot = create_data_loader(X_domain, y_domain, False, batch_size)
        y_pred_domain, _ = func_to_pred(data_loader_one_shot)
    else:
        X_domain, y_domain = preprocess_data(prepared_domain_text, tokenizer, punctuation_enc, segment_size)
        data_loader_one_shot = create_data_loader(X_domain, y_domain, False, batch_size)
        y_pred_domain, _ = func_to_pred(data_loader_one_shot)

    return prepared_domain_text, y_pred_domain


def capitalize(text: str) -> str:
    text = text.strip()
    if len(text) == 0:
        return ""

    text = text.capitalize()
    splitted_text = re.split("([.]\s*)", text)
    splitted_text = [substring.capitalize() for substring in splitted_text]
    capitalized_text = "".join(splitted_text)

    return capitalized_text


def cnt_punct(s):
    count = 0
    for i in range(0, len(s)):
        # Checks whether given character is a punctuation mark
        if s[i] in ('!', ",", "\'", ";", "\"", ".", "-", "?"):
            count = count + 1

    return count


def model_and_tokenizer_initialize(hyperparameters: dict):
    model_name = hyperparameters['model']['name_or_path']

    output_size = len(config.punctuation_encoding)

    tokenizer = BertWordPieceTokenizer(config.flavor + '/vocab.txt', lowercase=True)
    puncRec = nn.DataParallel(PuncRec(config).to(config.device))

    puncRec.load_state_dict(torch.load(get_path_to_checkpoint(), map_location=config.device), strict=False)

    puncRec.eval()

    return puncRec, tokenizer


def inference(input_text: str, bert_punc, tokenizer, hyperparameters: dict, batch_size: int = 2048):
    input_text = input_text.strip()

    func_to_pred = partial(predictions, bert_punc=bert_punc, device=config.device)

    prepared_domain_text, y_pred_domain = make_single_text_pred(input_text,
                                                                func_to_pred,
                                                                hyperparameters['segment_size'],
                                                                batch_size,
                                                                tokenizer,
                                                                punctuation_enc,
                                                                config.device
                                                                )
    print(y_pred_domain)

    res = right_decode_predictions(prepared_domain_text, y_pred_domain,
                                   tokenizer,
                                   punctuation_enc,
                                   hyperparameters["segment_size"])

    res = res[:len(input_text) + cnt_punct(res)]

    res = capitalize(res)
    return res


def prepare_hyperparameters():
    path = "models/release-candidate/hyperparameters.json"
    path = resource_filename(__name__, path)
    with open(path, 'r') as f:
        hyperparameters = json.load(f)

    return hyperparameters


def get_path_to_checkpoint():
    path = "models/release-candidate/model"
    path = resource_filename(__name__, path)
    return path


if __name__ == '__main__':
    BATCH_SIZE = 64

    hyperparameters = prepare_hyperparameters()

    puncRec, tokenizer = model_and_tokenizer_initialize(hyperparameters)

    input_text = "аның фикеренчә физик культураны һәм спортны популярлаштыруда волонтерларның роле зур хәзер " \
                 "татарстаннан бик күп волонтер сочига олимпиадага барырга әзерләнә очрашуда татарстанда волонтерлык " \
                 "хәрәкәтенең оешып җиткәнлеге күп тапкыр әйтелде республикада яшьләр турында закон кабул ителде анда " \
                 "иреклеләр хәрәкәтенә (добровольчество) ярдәм итү турында статья бар хәзер республикада 803 " \
                 "иреклеләр оешмасы эшли быел бездә 14-30 яшьтәге волонтерларның саны 19 мең кешегә арткан һәм 49 мең " \
                 "булган очрашуда катнашучылар фикеренчә волонтерлык эшчәнлеген кызыксындыру системасын киңәйтү " \
                 "турында уйларга кирәк дәүләт бүләге яки аерым бер билге стимул була ала дип саный тр иреклеләр " \
                 "хәрәкәте үсеше үзәге директоры анна синеглазова "

    print(input_text)

    punc_case_restored_text = inference(input_text, puncRec, tokenizer, hyperparameters, BATCH_SIZE)

    print(punc_case_restored_text)
