from getModels import getEnglishToHindiModel

from tqdm import tqdm


english_to_hindi = getEnglishToHindiModel()

with open("mt-output-hindi.txt" , "w") as hindi_output, open("data/english-validation.txt") as english_data:
    for sentence in tqdm(english_data.readlines() , desc = "translating"):
        mod_sent = sentence.strip()
        translation = english_to_hindi(mod_sent)[0]['translation_text']
        hindi_output.write(translation + "\n")
