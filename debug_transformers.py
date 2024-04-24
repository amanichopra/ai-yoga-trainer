
import wandb
import os
import numpy as np
import urllib.request as urllib
import pandas as pd
import random
import torch

from PIL import ImageDraw, ImageFont, Image
from datasets import load_dataset, load_metric
from transformers import ViTImageProcessor, ViTForImageClassification, TrainingArguments, Trainer


def process_example(example, processor):
    inputs = processor(example['image'], return_tensors='pt')
    inputs['labels'] = example['label']
    return inputs


def collate_func(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def main():
    # Add the code from train_vit.ipynb here
    IMGS_DATA_ROOT = './Yoga-82-Imgs'

    ds = load_dataset("imagefolder", data_dir=IMGS_DATA_ROOT)
    model_name_or_path = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name_or_path, input_data_format="channels_first")

    def transform(example_batch):
        # Make sure all of the images are in 'RGB' mode with 3 channels
        rgb_inputs = [(lambda x: x.convert('RGB'))(item) for item in example_batch['image']]

        # Take a list of PIL images and turn them to pixel values
        inputs = processor([x for x in rgb_inputs], return_tensors='pt')

        # Include labels
        inputs['labels'] = example_batch['label']
        return inputs
    
    # We would lose ~ 800 samples across the three splits which is less than 8% of the data
    # ds.filter(lambda x: x['image'].mode != 'RGB')

    transformed_ds = ds.with_transform(transform)

    labels = ds['train'].features['label'].names

    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="drive/MyDrive/AML_final_proj/vit-base-yoga82",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_func,
        compute_metrics=compute_metrics,
        train_dataset=transformed_ds["train"],
        eval_dataset=transformed_ds["validation"],
        tokenizer=processor,
    )

    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

if __name__ == "__main__":
    main()