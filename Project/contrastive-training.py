# %%
import os

os.environ['HF_HOME'] = "./hf/"

os.environ['WANDB_DISABLED'] = 'true'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




# %%
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
# for Mbart
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import BertGenerationEncoder , BertGenerationDecoder, BertTokenizerFast
from transformers import EncoderDecoderModel, EncoderDecoderConfig
import torch.nn.functional as F


import argparse


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device


argparser = argparse.ArgumentParser()


argparser.add_argument("--epochs" , "-e" , type = int , default = 1 , help = "Number of Epochs")


argparser.add_argument("--train_batch_size" , "-tbs" , type = int , default = 10 , help = "Train Batch Size")


argparser.add_argument("--validation_batch_size" , "-vbs" , type = int , default = 10 , help = "Validation Batch Size")

argparser.add_argument("--log_file" , "-lf" , type = str , default = 'log.txt' , help = "Log File")

argparser.add_argument("--save_path" , "-sp" , type = str , default = 'checkpoints/mBART' , help = "Save Path")

args = argparser.parse_args()


print("training batch : " , args.train_batch_size)
print("validation batch size : " , args.validation_batch_size)
print("epochs :" , args.epochs)






# %%
model_name = "google-bert/bert-base-multilingual-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

# %%
encoder_1 = BertGenerationEncoder.from_pretrained(model_name , bos_token_id=101, eos_token_id=102).to(device)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder_1 = BertGenerationDecoder.from_pretrained(model_name,add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102).to(device)

autoencoder_model_1 = EncoderDecoderModel(encoder=encoder_1,decoder=decoder_1).to(device)

autoencoder_model_1.config.decoder_start_token_id = tokenizer.cls_token_id
autoencoder_model_1.config.pad_token_id = tokenizer.pad_token_id
autoencoder_model_1.config.vocab_size = autoencoder_model_1.config.decoder.vocab_size



# %%
encoder_2 = BertGenerationEncoder.from_pretrained(model_name , bos_token_id=101, eos_token_id=102).to(device)
# add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
decoder_2 = BertGenerationDecoder.from_pretrained(model_name,add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102).to(device)


autoencoder_model_2 = EncoderDecoderModel(encoder=encoder_2,decoder=decoder_2).to(device)

autoencoder_model_2.config.decoder_start_token_id = tokenizer.cls_token_id
autoencoder_model_2.config.pad_token_id = tokenizer.pad_token_id
autoencoder_model_2.config.vocab_size = autoencoder_model_2.config.decoder.vocab_size



# %%
optimizer = Adam(list(autoencoder_model_1.parameters()) + list(autoencoder_model_2.parameters()), lr=1e-5)

# %%
def contrastive_loss(outputs1, outputs2, margin=1):
    # Calculate Euclidean distance between the encoded representations
    distance = F.pairwise_distance(outputs1, outputs2)
    # Calculate contrastive loss
    loss_contrastive = torch.mean((margin - distance) ** 2)  # Squared hinge loss
    return loss_contrastive

# %%
import datasets
from datasets import load_dataset

# datasets.set_progress_bar_enabled(False)
datasets.logging.disable_progress_bar

dataset = load_dataset("cfilt/iitb-english-hindi")

# %%
from datasets import Dataset
def generate_dataset(dataset , split):
    filtered_dataset = dataset[split]['translation']
    english_dataset = [data['en'] for data in filtered_dataset]
    hindi_dataset = [data['hi'] for data in filtered_dataset]
    dataset_size = min(10000 , len(english_dataset))


    print("Total Dataset length : " , len(english_dataset))
    print("Trimmed length :" , dataset_size)


    english_dataset = english_dataset[:dataset_size]
    hindi_dataset = hindi_dataset[:dataset_size]
    data_dictionary = {
        "english" : english_dataset,
        "hindi" : hindi_dataset
    }
    return Dataset.from_dict(data_dictionary)


# %%
def get_dataset_as_list(dataset, split):
    filtered_dataset = dataset[split]['translation']
    english_dataset = [data['en'] for data in filtered_dataset]
    hindi_dataset = [data['hi'] for data in filtered_dataset]
    return english_dataset, hindi_dataset


# %%
train_dataset = generate_dataset(dataset, "train")
train_dataset
            

# %%
test_dataset = generate_dataset(dataset , "test")
test_dataset

# %%
validation_dataset = generate_dataset(dataset , "validation")
validation_dataset

# %%
from transformers import DataCollatorForSeq2Seq
data_collector = DataCollatorForSeq2Seq(tokenizer=tokenizer)

# %%
from torch.utils.data import DataLoader


# %%
def tokenize_dataset(example):
    model_inputs = tokenizer(example["english"], max_length=512, truncation=True , return_tensors='pt')
    labels = tokenizer(example["hindi"], max_length=512, truncation=True, return_tensors='pt')
    model_inputs["hindi_input_ids"] = labels["input_ids"]
    model_inputs['hindi_attn_mask'] = labels['attention_mask']
    return model_inputs


def tokenize_dataset_english(example):
    model_inputs = tokenizer(example["english"], max_length=512, truncation=True)
    model_inputs['labels'] = model_inputs['input_ids']
    return model_inputs


def tokenize_dataset_hindi(example):
    model_inputs = tokenizer(example["hindi"], max_length=512, truncation=True)
    model_inputs['labels'] = model_inputs['input_ids']
    return model_inputs

# %%
from transformers import get_scheduler
from tqdm import tqdm
import time


def train(autoencoder_model_1 = autoencoder_model_1 , autoencoder_model_2 = autoencoder_model_2, epochs=1):
    autoencoder_model_1.train()
    autoencoder_model_2.train()
    for epoch in tqdm(range(epochs) , desc='epochs'):
        total_autoencoder1_loss = 0
        total_autoencoder2_loss = 0
        total_contrastive_loss = 0

        num_training_steps = epoch * len(train_dataset)
        lr_scheduler = get_scheduler(
                "linear",
                optimizer = optimizer,
                num_warmup_steps= 100,
                num_training_steps = num_training_steps
            )

        steps = 0
        batch_size = args.train_batch_size

        for idx in tqdm(range(0, len(train_dataset['english']) , batch_size) , desc = "training"):
            steps+=1

            english_batch = train_dataset['english'][idx:idx+batch_size]
            hindi_batch = train_dataset['hindi'][idx:idx+batch_size]
            data_dictionary = {
                "english" : english_batch,
                "hindi" : hindi_batch
            }

            batch =  Dataset.from_dict(data_dictionary) 
            english_tokens = batch.map(tokenize_dataset_english).remove_columns(['english' , 'hindi' ])
            hindi_tokens = batch.map(tokenize_dataset_hindi).remove_columns(['english' , 'hindi' ])


            english_dataloader = DataLoader(english_tokens ,
                                        batch_size = batch_size,
                                        collate_fn = data_collector
                                        )

            hindi_dataloader = DataLoader(hindi_tokens ,
                                        batch_size = batch_size,
                                        collate_fn = data_collector
                                        )
            
            for eng_batch,hin_batch in zip(english_dataloader,hindi_dataloader):
                # for english
                english_labels = eng_batch['labels'].to(device)
                english_labels = torch.where(english_labels != -100, english_labels, tokenizer.pad_token_id)
                english_inputs = english_labels
                english_attention_mask = eng_batch['attention_mask'].to(device)

                value_1 = autoencoder_model_1(input_ids = english_inputs , labels = english_labels , attention_mask = english_attention_mask)
                loss_1 = value_1.loss


                encoder_1_output = value_1.encoder_last_hidden_state.mean(dim=1)
                # print("encoder_1_shape" , encoder_1_output.shape)
                



                # for hindi

                hindi_labels = hin_batch['labels'].to(device)
                hindi_labels = torch.where(hindi_labels != -100, hindi_labels, tokenizer.pad_token_id)
                hindi_inputs = hindi_labels
                hindi_attention_mask = hin_batch['attention_mask'].to(device)


                value_2 = autoencoder_model_2(input_ids = hindi_inputs , labels = hindi_labels , attention_mask = hindi_attention_mask)
                loss_2 = value_2.loss


                encoder_2_output = value_2.encoder_last_hidden_state.mean(dim=1)

                # print("encoder_2_shape" , encoder_2_output.shape)


                contrastive_loss_calc = contrastive_loss(encoder_1_output,encoder_2_output)

                # print("contranstive loss is:" , contrastive_loss_calc)


                loss = loss_1 + loss_2 + contrastive_loss_calc


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()




                total_autoencoder1_loss += loss_1.item()
                total_autoencoder2_loss += loss_2.item()
                total_contrastive_loss += contrastive_loss_calc.item()



                if steps % 10 == 0:
                    print("Total Loss " , loss.item() ,f"Contrastive Loss: {contrastive_loss_calc.item()}")
                    with open(args.log_file , "a") as log:
                        log_message = f"""
                        Epoch {epoch} Step: {steps} , Time = {time.time()}
                        f"Autoencoder , 1 Loss for batch: {loss_1.item()}
                        f"Autoencoder 2 Loss for batch: {loss_2.item()}
                        f"Contrastive Loss for batch: {contrastive_loss_calc.item()}
                        Total Loss for batch: {loss.item()} 
                        *****************
                        """
                        log.write(log_message)
                # save checkpoint every thousand steps
                if (steps+1) % 100 == 0:
                    model_1_checkpoint = f"{args.save_path}/checkpoint/autoencoder_model_1_checkpoint_{epoch}_{steps+1}"
                    model_2_checkpoint = f"{args.save_path}/checkpoint/autoencoder_model_2_checkpoint_{epoch}_{steps+1}"
                    autoencoder_model_1.save_pretrained(model_1_checkpoint)
                    autoencoder_model_2.save_pretrained(model_2_checkpoint)



            # Print epoch statistics
        with open(args.log_file , "a") as log:
            print("Total Loss:")
            print(f"Epoch {epoch + 1}:")
            print(f"Total Autoencoder 1 Loss: {total_autoencoder1_loss / len(train_dataset) * batch_size}")
            print(f"Total Autoencoder 2 Loss: {total_autoencoder2_loss / len(train_dataset) * batch_size}")
            print(f"Total Contrastive Loss: {total_contrastive_loss / len(train_dataset) * batch_size}")

            log_message = f"""
Total Loss for Epoch {epoch + 1}:
    Total Autoencoder 1 Loss: {total_autoencoder1_loss / len(train_dataset) * batch_size}  
    Total Autoencoder 2 Loss: {total_autoencoder2_loss / len(train_dataset) * batch_size} 
    Total Contrastive Loss: {total_contrastive_loss / len(train_dataset) * batch_size}         
            """






                # print("Total Loss :  ", loss.item())




# %%
train(epochs=args.epochs)



autoencoder_model_1.save_pretrained(f"{args.save_path}/autoencoder_model_1")
autoencoder_model_2.save_pretrained(f"{args.save_path}/autoencoder_model_2")




# %%
# Inference
def translate(src_sentence):
    print(src_sentence)
    tokens = tokenizer(src_sentence, return_tensors='pt').to(device)
    # print(tokens)
    encoded_src = autoencoder_model_1.encoder(input_ids = tokens.input_ids , attention_mask = tokens.attention_mask).last_hidden_state
    # print(encoded_src)
    generated_tgt = autoencoder_model_2.decoder.generate(encoder_hidden_states=encoded_src)
    # print(generated_tgt)
    decoded_tgt = tokenizer.decode(generated_tgt[0], skip_special_tokens=True)
    return decoded_tgt

# %%
# translate("are you feeling well")

# %%



