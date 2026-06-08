from scripts.gen_berts import *
from pathlib import Path
import re
import json

workspace_path = Path('~/Development/digital-latin/2026-digital-latin-iii/wsi/python/workspace').expanduser()
result_base_path = workspace_path / 'bert'
result_base_path.mkdir(exist_ok=True)

bert=LatinBERT(tokenizerPath=str(Path(__file__).parent.parent.parent / 'models/subword_tokenizer_latin/latin.subword.encoder'), bertPath=str(Path(__file__).parent.parent.parent / 'models/latin_bert/'))

# with Path('/home/giulio/Development/digital-latin/2026-digital-latin-iii/wsi/python/workspace/alim/437-scriptum_super_libros_sententiarum,_ii/text.txt').open('r') as fp:
#     sentence = fp.read()
#     sentences = [sentence]
#     bert_sents=bert.get_berts(sentences)
#     for sent in bert_sents:
#         for (token, bert) in sent:
#             print("%s\t%s" % ( token, ' '.join(["%.5f" % x for x in bert])))

def search(term : str, context_sentences_window_radius : int = 0):
    result_path = result_base_path / f'bert_search_term__{term}__paragraphs_context_window__{context_sentences_window_radius}.json'
    if not result_path.exists():
        result_obj = {}
        for dir in (workspace_path / 'alim').iterdir():
            if not dir.is_dir():
                continue
            for paragraph_path in sorted((dir / 'paragraphs').iterdir()):
                with paragraph_path.open('r') as paragraph_fp:
                    text = paragraph_fp.read()
                    if term.lower() in text.lower():
                        if not dir.name in result_obj:
                            result_obj[dir.name] = {}
                        paragraph_start_index = max(0, int(paragraph_path.name[10:16]) - context_sentences_window_radius)
                        paragraph_stop_index = int(paragraph_path.name[10:16]) + context_sentences_window_radius + 1
                        sents = []
                        for paragraph_index in range(paragraph_start_index, paragraph_stop_index):
                            sentence_path = dir / 'paragraphs' / f'paragraph_{paragraph_index:06}.txt'
                            if sentence_path.exists():
                                with sentence_path.open('r') as sentence_fp:
                                    sents.append(sentence_fp.read())
                        result_obj[dir.name][paragraph_path.name] = {'sentences' : sents, 'bert' : []}
                        bert_sents = bert.get_berts(sents)
                        for sent in bert_sents:
                            for (token, embedding) in sent:
                                result_obj[dir.name][paragraph_path.name]['bert'].append({'token': token, 'embedding' : embedding.tolist()})
        with result_path.open('w') as result_fp:
            json.dump(result_obj, result_fp, indent=4)

search('maneries', 3)