import torch
from transformers import MBartForConditionalGeneration, MBartTokenizer
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import BitsAndBytesConfig
import numpy as np
# import peft and Lora
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from peft import PeftModel
from transformers import pipeline




def getEnglishToHindiModel():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBartTokenizer.from_pretrained(model_name , use_fast = True)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    ft_model = PeftModel.from_pretrained(model, "models/mBART-fine-tuned-eng-hi",torch_dtype=torch.float16,is_trainable=False)

    translator = pipeline("translation", model=ft_model,
                    src_lang = "en_XX" , tgt_lang= "hi_IN",
                    tokenizer=tokenizer)

    return translator




def getHindiToEnglishModel():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBartTokenizer.from_pretrained(model_name , use_fast = True)
    model = MBartForConditionalGeneration.from_pretrained(model_name)

    ft_model = PeftModel.from_pretrained(model, "models/mBART-fine-tuned-hi-en",torch_dtype=torch.float16,is_trainable=False)

    translator = pipeline("translation", model=ft_model,
                    src_lang = "hi_IN" , tgt_lang= "eng_XX",
                    tokenizer=tokenizer)

    return translator







# english = getEnglishToHindiModel()
# hindi = getEnglishToHindiModel()



# print(english("Hi how are you"))
# print(hindi("आप कैसे हैं"))