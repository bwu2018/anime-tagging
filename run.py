import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import torch
import glob
# import pytorch_lightning as pl
from huggingface_hub import HfApi, Repository
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics import Accuracy
from transformers import ViTFeatureExtractor, ViTForImageClassification
# from pytorch_lightning.callbacks import ModelCheckpoint

feature_extractor = ViTFeatureExtractor.from_pretrained('./outputs/checkpoint-1761')
model = ViTForImageClassification.from_pretrained('./outputs/checkpoint-1761')

id2label = {
    "0": "1girl",
    "1": "ahoge",
    "10": "blush",
    "11": "boots",
    "12": "bow",
    "13": "breasts",
    "14": "brown_eyes",
    "15": "brown_hair",
    "16": "cat_ears",
    "17": "catgirl",
    "18": "cleavage",
    "19": "closed_eyes",
    "2": "animal_ears",
    "20": "d.gray-man",
    "21": "detached_sleeves",
    "22": "dress",
    "23": "elbow_gloves",
    "24": "flat_chest",
    "25": "flower",
    "26": "glasses",
    "27": "gloves",
    "28": "green_eyes",
    "29": "green_hair",
    "3": "aqua_hair",
    "30": "hair_bow",
    "31": "hair_ornament",
    "32": "hair_ribbon",
    "33": "hat",
    "34": "hatsune_miku",
    "35": "headphones",
    "36": "highres",
    "37": "hoshino_katsura",
    "38": "itou_noiji",
    "39": "japanese_clothes",
    "4": "bad_id",
    "40": "jewelry",
    "41": "kagamine_len",
    "42": "kimono",
    "43": "legs",
    "44": "long_hair",
    "45": "male",
    "46": "megurine_luka",
    "47": "midriff",
    "48": "monochrome",
    "49": "multiple_girls",
    "5": "bikini",
    "50": "nagato_yuki",
    "51": "nail_polish",
    "52": "navel",
    "53": "necktie",
    "54": "open_mouth",
    "55": "original",
    "56": "pantyhose",
    "57": "pink_hair",
    "58": "ponytail",
    "59": "purple_eyes",
    "6": "black_hair",
    "60": "purple_hair",
    "61": "red_eyes",
    "62": "red_hair",
    "63": "ribbon",
    "64": "scan",
    "65": "scarf",
    "66": "school_uniform",
    "67": "seifuku",
    "68": "shoes",
    "69": "short_hair",
    "7": "blonde_hair",
    "70": "sitting",
    "71": "skirt",
    "72": "sky",
    "73": "smile",
    "74": "socks",
    "75": "solo",
    "76": "suzumiya_haruhi_no_yuuutsu",
    "77": "swimsuit",
    "78": "sword",
    "79": "tagme",
    "8": "blue_eyes",
    "80": "tail",
    "81": "thigh-highs",
    "82": "thigh_highs",
    "83": "thighhighs",
    "84": "touhou",
    "85": "trap",
    "86": "twintails",
    "87": "uniform",
    "88": "vector",
    "89": "very_long_hair",
    "9": "blue_hair",
    "90": "vocaloid",
    "91": "wallpaper",
    "92": "water",
    "93": "weapon",
    "94": "white",
    "95": "white_hair",
    "96": "wings",
    "97": "wink",
    "98": "yellow_eyes",
    "99": "zettai_ryouiki"
  }
label2id = {
    "1girl": "0",
    "ahoge": "1",
    "animal_ears": "2",
    "aqua_hair": "3",
    "bad_id": "4",
    "bikini": "5",
    "black_hair": "6",
    "blonde_hair": "7",
    "blue_eyes": "8",
    "blue_hair": "9",
    "blush": "10",
    "boots": "11",
    "bow": "12",
    "breasts": "13",
    "brown_eyes": "14",
    "brown_hair": "15",
    "cat_ears": "16",
    "catgirl": "17",
    "cleavage": "18",
    "closed_eyes": "19",
    "d.gray-man": "20",
    "detached_sleeves": "21",
    "dress": "22",
    "elbow_gloves": "23",
    "flat_chest": "24",
    "flower": "25",
    "glasses": "26",
    "gloves": "27",
    "green_eyes": "28",
    "green_hair": "29",
    "hair_bow": "30",
    "hair_ornament": "31",
    "hair_ribbon": "32",
    "hat": "33",
    "hatsune_miku": "34",
    "headphones": "35",
    "highres": "36",
    "hoshino_katsura": "37",
    "itou_noiji": "38",
    "japanese_clothes": "39",
    "jewelry": "40",
    "kagamine_len": "41",
    "kimono": "42",
    "legs": "43",
    "long_hair": "44",
    "male": "45",
    "megurine_luka": "46",
    "midriff": "47",
    "monochrome": "48",
    "multiple_girls": "49",
    "nagato_yuki": "50",
    "nail_polish": "51",
    "navel": "52",
    "necktie": "53",
    "open_mouth": "54",
    "original": "55",
    "pantyhose": "56",
    "pink_hair": "57",
    "ponytail": "58",
    "purple_eyes": "59",
    "purple_hair": "60",
    "red_eyes": "61",
    "red_hair": "62",
    "ribbon": "63",
    "scan": "64",
    "scarf": "65",
    "school_uniform": "66",
    "seifuku": "67",
    "shoes": "68",
    "short_hair": "69",
    "sitting": "70",
    "skirt": "71",
    "sky": "72",
    "smile": "73",
    "socks": "74",
    "solo": "75",
    "suzumiya_haruhi_no_yuuutsu": "76",
    "swimsuit": "77",
    "sword": "78",
    "tagme": "79",
    "tail": "80",
    "thigh-highs": "81",
    "thigh_highs": "82",
    "thighhighs": "83",
    "touhou": "84",
    "trap": "85",
    "twintails": "86",
    "uniform": "87",
    "vector": "88",
    "very_long_hair": "89",
    "vocaloid": "90",
    "wallpaper": "91",
    "water": "92",
    "weapon": "93",
    "white": "94",
    "white_hair": "95",
    "wings": "96",
    "wink": "97",
    "yellow_eyes": "98",
    "zettai_ryouiki": "99"
  }


def prediction(img_path):
   im=Image.open(img_path)
   encoding = feature_extractor(images=im, return_tensors="pt")
   encoding.keys()
   pixel_values = encoding['pixel_values']
   outputs = model(pixel_values)
   result = outputs.logits.softmax(1).argmax(1)
   new_result = result.tolist() 
   for i in new_result:
     return(id2label[str(i)])

root_dir = "../anime-tagging-dataset/anime-dataset/"
example_path = root_dir + "1girl/1.jpg"
blue_hair = root_dir + "blue_hair/909.jpg"
closed_eyes = root_dir + "closed_eyes/1127.jpg"
male = root_dir + "male/286.jpg"
print(prediction(example_path))
print(prediction(blue_hair))
print(prediction(closed_eyes))
print(prediction(male))