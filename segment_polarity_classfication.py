import os

from datasets import load_dataset, concatenate_datasets
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import pandas as pd


def training(args):
    dataset_imdb = load_dataset("csv", data_files=['./spot-data/data/spot-imdb-edus.csv'])
    dataset_yelp = load_dataset("csv", data_files=['./spot-data/data/spot-yelp13-edus.csv'])
    dataset = concatenate_datasets([dataset_imdb['train'], dataset_yelp['train']])

    tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

    model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity",
                                                          num_labels=2)

    def encode(batch):
        return tokenizer(batch['text'], truncation=True)

    dataset = dataset.map(encode, batched=True, batch_size=args.train_batch_size)

    dataset = dataset.train_test_split(test_size=0.2)

    # def compute_metrics(pred):
    #     labels = pred.label_ids
    #     # index at zero if output_hidden_states=True in instantiation of model
    #     preds = pred.predictions[0].argmax(-1)
    #     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    #     acc = accuracy_score(labels, preds)
    #     classification_rpt = classification_report(labels, preds, target_names=list(label_encoder.label_space),
    #                                                zero_division=0)
    #     return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall,
    #             "classification_report": classification_rpt}

    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=args.learning_rate,
        lr_scheduler_type='linear',
        group_by_length=args.group_by_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
    )

    # trainer.train()


def inference(args):
    model = AutoModelForSequenceClassification.from_pretrained(
        "malsalih/autotrain-segment-polarity-classification-1063636922", use_auth_token='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')

    tokenizer = AutoTokenizer.from_pretrained("malsalih/autotrain-segment-polarity-classification-1063636922",
                                              use_auth_token='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')

    def get_input_files(folder_path_pattern):
        folder = folder_path_pattern[:-3]
        path_list = []
        for file in os.listdir(folder):
            path_list.append(folder_path_pattern.format(file))
        return path_list

    input_files = get_input_files(args.folder_path_pattern)

    dataset = load_dataset("text", data_files=input_files[:50])

    print(dataset['train']['text'])

    def encode(batch):
        return tokenizer(batch['text'], truncation=True)

    dataset = dataset.map(encode, batched=True, batch_size=args.train_batch_size)

    training_args = TrainingArguments(
        output_dir=args.output_data_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_dir=f"{args.output_data_dir}/logs",
        learning_rate=args.learning_rate,
        lr_scheduler_type='linear',
        group_by_length=args.group_by_length,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        # compute_metrics=compute_metrics,
        # train_dataset=dataset['train'],
        eval_dataset=dataset['train'],
        tokenizer=tokenizer,
    )
    results = trainer.predict(test_dataset=dataset['train']).predictions
    values = {text: logits.argmax() for text, logits in zip(dataset['train']['text'], results)}
    values = pd.Series(values)
    negative_segments = values[values == 0]
    positive_segments = values[values == 1]
    negative_segments.to_csv('./review_data/negative_segments/negative_segments.csv')
    positive_segments.to_csv('./review_data/positive_segments/positive_segments.csv')
    print("Finished splitting files into polarity")



def main():
    parser = argparse.ArgumentParser("polarity classification over segments")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--learning_rate", type=float, default=4e-5)
    parser.add_argument("--save_steps", type=int, default=4000)
    parser.add_argument("--group_by_length", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=False)
    parser.add_argument('--output_data_dir', type=str, default='./results')
    parser.add_argument('--folder_path_pattern', type=str, default='./review_data/segments/{}')
    parser.add_argument("--inference", type=bool, default=True)
    args = parser.parse_args()

    if args.inference:
        inference(args)
    else:
        training(args)


if __name__ == '__main__':
    main()
