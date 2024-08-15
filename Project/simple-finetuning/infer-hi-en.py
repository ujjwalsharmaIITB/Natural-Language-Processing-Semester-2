from getModels import getHindiToEnglishModel

from tqdm import tqdm


hindi_to_english = getHindiToEnglishModel()

with open("mt-output-english.txt" , "w") as english_output, open("data/hindi-validation.txt") as hindi_data:
    for sentence in tqdm(hindi_data.readlines() , desc = "translating"):
        mod_sent = sentence.strip()
        translation = hindi_to_english(mod_sent)[0]['translation_text']
        english_output.write(translation + "\n")
