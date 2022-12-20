import re

from string import punctuation

from pymystem3 import Mystem
from nltk.corpus import stopwords


class FinePreprocessorText():
    def __init__(self):
        self.mystem = Mystem(grammar_info=True, disambiguation=True)
        self.ru_stopwords = stopwords.words('russian')
        self.en_stopwords = stopwords.words('english')
    
    def __call__(self, text):
        text = re.sub(r' ?\w*tokenoid', '', text) # remove tokenoid
        
        for tag in re.finditer(r'([А-ЯЁ][а-яё]+)([А-ЯЁ][а-яё]+)', text): # First_nameSecond_name tag
            text = re.sub(tag.group(0), ' '.join(tag.groups()), text)
        
        text = ' '.join([word for word in re.findall(r'\w{3,}', text) if not word.isdigit()])
        
        fine_text = []
        for analysis in self.mystem.analyze(text):
            if 'analysis' in analysis:
                if analysis['analysis']:
                    lexes = []

                    # check first and second name
                    for lex in analysis['analysis']:
                        if 'имя' in lex['gr'] or 'фам' in lex['gr']:
                            lexes.append(lex['lex'].capitalize())

                    # check more context-independent lex
                    context_lex, independent = None, -1
                    if not lexes:
                        for lex in analysis['analysis']:
                            if lex['wt'] > independent:
                                context_lex, independent = lex['lex'], lex['wt']

                        fine_text.append(context_lex)
                    else:
                        fine_text += lexes
                else:
                    fine_text.append(analysis['text'].lower())
            elif 'text' in analysis and analysis['text'] != ' ':
                unrecognized_text = analysis['text'].strip()
                if unrecognized_text not in punctuation and not unrecognized_text.isdigit():
                    fine_text.append(unrecognized_text.lower())
            else:
                pass
        
        fine_text = ' '.join([lex for lex in fine_text \
                              if lex not in self.ru_stopwords \
                              and lex not in self.en_stopwords])

        return fine_text
