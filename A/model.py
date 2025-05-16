# Model definition and training/test setup for legal summarization

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments,
    TrainerCallback
)
from transformers import DataCollatorForSeq2Seq
from A.utils import WeightedLossTrainer, DataCollatorWithWeights, SummaCEarlyStoppingCallback, CurriculumCallback
import evaluate
from summac.model_summac import SummaCConv

class LegalSummarizer:
    """
    Wrapper class for training and testing a legal summarization model using CodeT5.
    """
    def __init__(self):
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-small")

        # Add task-specific custom control and placeholder tokens
        self.tokenizer.add_tokens([
            "[SEC=FEW]", "[SEC=MANY]",
            "[HIGHLIGHT]", "[/HIGHLIGHT]",
            "[ECP=LOW]", "[ECP=MEDIUM]", "[ECP=HIGH]",
            "[SENT_ECP=LOW]", "[SENT_ECP=MEDIUM]", "[SENT_ECP=HIGH]",
            "@ORG_1@", "@ORG_2@", "@ORG_3@",
            "@PERSON_1@", "@PERSON_2@",
            "@DATE_1@", "@DATE_2@",
            "@GPE_1@", "@LAW_1@", "@LOC_1@"
        ])
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Define training arguments for the Seq2SeqTrainer
        self.training_args = Seq2SeqTrainingArguments(
            output_dir="./codet5-dual-summary",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            save_steps=500,
            logging_steps=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            learning_rate=2.4e-5,
            lr_scheduler_type="linear",
            warmup_steps=300,
            weight_decay=0.1,
            label_smoothing_factor=0.1,
            predict_with_generate=True,
            max_grad_norm=1.0,
            save_total_limit=2,
            load_best_model_at_end=True,
            greater_is_better=True,
            report_to="none",
            generation_max_length=384,
            generation_num_beams=5,
        )

    def train(self, train_dataset, val_dataset):
        """
        Trains the model using custom weighted loss and optional callbacks foro SummaC and ROUGE.
        """
        # Update model config with decoding parameters
        self.model.config.max_length = 384
        self.model.config.num_beams = 5
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        data_collator = DataCollatorWithWeights(tokenizer=self.tokenizer, model=self.model)

        # Small subset for potential early stopping (currently commented)
        eval_subset = val_dataset.select(range(50))

        trainer = WeightedLossTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[
                # Optional: enable SummaC-based early stopping
                # SummaCEarlyStoppingCallback(
                #     tokenizer=self.tokenizer,
                #     eval_dataset=eval_subset,
                #     patience=5,
                #     min_delta=0.5,
                #     eval_every_n_epochs=2,
                # )
            ]
        )

        # Optional: curriculum learning support
        # trainer.callbacks.append(CurriculumCallback(trainer_ref=trainer))

        trainer.train()

        # Optional: save trained model to dataset directory
        # self.model.save_pretrained("./codet5-dual-summary/final_model")
        # self.tokenizer.save_pretrained("./codet5-dual-summary/final_model")

        return "DONE"

    def test(self, test_dataset):
        """
        Runs inference on the test dataset and prints ROUGE-L and SummaC scores.
        """
        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        generated = []
        references = []

        for example in test_dataset:
            input_text = example["input"]
            ref_summary = example["label"]

            # Tokenize input and run generation
            tokenized = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model.generate(
                    **tokenized,
                    max_length=384,
                    num_beams=5,
                    no_repeat_ngram_size=3
                )
            gen_summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
            generated.append(gen_summary)
            references.append(ref_summary)

        # Compute ROUGE-L score
        rouge = evaluate.load("rouge")
        rouge_result = rouge.compute(predictions=generated, references=references)

        # Compute factual consistency with SummaC
        summac = SummaCConv(
            models=["mnli"],
            granularity="sentence",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        summac_score = summac.score(references, generated, aggregation="mean")["scores"][0]

        score_str = f"ROUGE-L: {rouge_result['rougeL']*100:.2f} | SummaC: {summac_score*100:.2f}"
        print("\n=== TEST RESULTS ===")
        print(score_str)
        return score_str
