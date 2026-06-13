from scripts.gen_berts import *
from pathlib import Path
import pickle
import re
import json
import traceback

workspace_path = Path('~/Development/digital-latin/2026-digital-latin-iii/wsi/workspace').expanduser()
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

def path_comparator(path : Path):
    m = re.search(r'\d+', path.name)
    if m:
        return int(m.group())
    return -1

def search(terms : list[str]):

    print (f'\n\nSEARCHING FOR TERMS: {",".join(terms)}')

    result_path = result_base_path / f'bert_search_term__{"-".join(terms)}.json'

    # per-document incremental job
    if result_path.exists():
        with result_path.open('r') as result_fp:
            result_obj = json.load(result_fp)
    else:
        result_obj = {}

    for dir_path in sorted((workspace_path / 'alim').iterdir(), key = path_comparator):
        
        if not dir_path.is_dir():
            continue

        if dir_path.name in result_obj:
            print (f'\nSkipping already present {dir_path.name}...')
            continue

        with (dir_path / 'text.txt').open('r') as text_fp:
            text = text_fp.read()

        text_matches = False
        for term in terms:
            if term in text.lower():
                text_matches = True
                break

        if text_matches:

            print (f'\nProcessing {dir_path.name}...')
            
            bert_cache_path = dir_path / 'bert_cache.pyc'
            if bert_cache_path.exists():
                print('\trestoring bert data from cache')
                with bert_cache_path.open('rb') as bert_cache_fp:
                    bert_sents = pickle.load(bert_cache_fp)
            else:
                try:
                    bert_sents = bert.get_berts([text])
                except RuntimeError as e:
                    print (e)
                    traceback.print_exc()
                    continue

                with bert_cache_path.open('wb') as bert_cache_fp:
                    pickle.dump(bert_sents, bert_cache_fp)

            result_obj[dir_path.name] = {'sentences' : []}
            term_occurrence_count = 0
            for sent_index, sent in enumerate(bert_sents):
                sentence = {'index' : sent_index, 'term_indexes' : []}
                term_found = False
                term_pos = 0
                embeddings = []
                for (token, embedding) in sent:
                    embeddings.append({'token': token, 'embedding' : embedding.tolist()})
                    if token in terms:
                        term_found = True
                        sentence['term_indexes'].append(term_pos)
                    term_pos += 1
                if term_found:
                    sentence['embeddings'] = embeddings
                    result_obj[dir_path.name]['sentences'].append(sentence)

            # save partial result
            with result_path.open('w') as result_fp:
                json.dump(result_obj, result_fp, indent=4)

            print ('done!')

for term in ['appositio','appositione','appositionem','appositiones',
             'appositionibus','maneries','maneriei','maneriebus',
             'mutatio','mutatione','mutationem','mutationes',
             'terminatio','terminationes','terminatione','terminationem','terminationibus','terminationis',
             'dispositio','dispositione','dispositionem','dispositionis','dispositiones','dispositionibus',
             'prologus','prologum','prologi',
             'dictamen','dictaminis','dictamine','dictaminum','dictaminibus',
             'color','colore','colores','coloris','coloribus','colorem','colorum',
             'modo','modum','modis','modus','modi','modos','modorum',
             ]:
    search([term])

