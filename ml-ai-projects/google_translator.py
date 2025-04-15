from googletrans import Translator

translator = Translator()

text = "Hello, how are you?"
translated = translator.translate(text, dest='fr')  # French
print("Translated:", translated.text)
