from googletrans import Translator
'''
def BackTranslation(text, original_lang, via_lang):
    translator = Translator()
    return translator.translate(translator.translate(text, dest=via_lang).text, dest=original_lang).text
'''

text = 'the wheel could be better , would you accept 520 ?'
translator = Translator()
print(text+ '\n')

tr = translator.translate(text, dest='ja').text
print(tr + '\n')

rev = translator.translate(tr, dest='en').text
print(rev) 