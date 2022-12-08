import re
import requests
import warnings


from pymystem3 import Mystem
from nltk.corpus import stopwords


class MystemTokenizer:
    def __init__(self, weight_thr=0.9, grammeme=True):       
        self.weight_thr = weight_thr
        self.grammeme = grammeme
        self.m = Mystem(grammar_info=grammeme)

        self.stopwords = frozenset(stopwords.words('russian'))

        url = 'https://raw.githubusercontent.com/akutuzov/universal-pos-tags/4653e8a9154e93fe2f417c7fdb7a357b7d6ce333/ru-rnc.map'
        r = requests.get(url, stream=True)
        if r.ok:
            self.mystem2upos = {k: v for k, v in re.findall(r'([A-Z]+)\s+([A-Z]+)', r.text)}
        else:
            warnings.warn('Failed to load Mystem tag conversion table to Universal Tags.')

            self.mystem2upos = {'A': 'ADJ',
                                'ADV': 'ADV',
                                'ADVPRO': 'ADV',
                                'ANUM': 'ADJ',
                                'APRO': 'DET',
                                'COM': 'ADJ',
                                'CONJ': 'SCONJ',
                                'INTJ': 'INTJ',
                                'NONLEX': 'X',
                                'NUM': 'NUM',
                                'PART': 'PART',
                                'PR': 'ADP',
                                'S': 'NOUN',
                                'SPRO': 'PRON',
                                'UNKN': 'X',
                                'V': 'VERB'}
    
    def __call__(self, text):
        tags = []
        for w in self.m.analyze(text):
            if 'analysis' in w and w['analysis']:
                tag = self._extract(w['analysis'][0])
                
                if tag is not None:
                    tags.append(tag)
        
        return tags
    
    def _extract(self, analysis):
        if analysis['wt'] > self.weight_thr and \
        analysis['lex'] not in self.stopwords and len(analysis['lex']) > 2:
            if self.grammeme:
                gr = re.match(r'^(\w+)[=,]', analysis['gr']).group(1)
                return analysis['lex'] + '_' + self.mystem2upos[gr]
            else:
                return analysis['lex']
