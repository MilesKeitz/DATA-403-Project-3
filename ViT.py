import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    from datasets import load_dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        TrainingArguments,
        Trainer,
        DefaultDataCollator,
        set_seed,
        EarlyStoppingCallback,
    )
    import evaluate
    import numpy as np
    import torch
    import torchvision.transforms as T
    import marimo as mo
    import csv
    import os
    import json

    set_seed(0)
    return (
        AutoImageProcessor,
        AutoModelForImageClassification,
        DefaultDataCollator,
        EarlyStoppingCallback,
        T,
        Trainer,
        TrainingArguments,
        csv,
        evaluate,
        json,
        load_dataset,
        np,
        os,
        torch,
    )


@app.cell
def _(torch):
    HAS_CUDA = torch.cuda.is_available()
    HAS_XPU = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())

    DEVICE = "cuda" if HAS_CUDA else ("xpu" if HAS_XPU else "cpu")
    BF16 = bool(
        HAS_XPU or (HAS_CUDA and torch.cuda.get_device_capability(0)[0] >= 8)
    )
    FP16 = bool(HAS_CUDA and not BF16)
    PIN_MEM = bool(HAS_CUDA)

    if HAS_XPU:
        # hacky, but gets rid of errors
        torch.Tensor.double = torch.Tensor.float
        torch.float64 = torch.double = torch.float32
    # print(DEVICE)
    # torch.set_default_device(DEVICE)
    return (PIN_MEM,)


@app.cell
def _(load_dataset):
    # 1) Load folder dataset and make a stratified train/val split
    raw = load_dataset(
        "imagefolder",
        data_files={"train": ["Data/Alex/**", "Data/Kelly/**"]},
        split="train",
    )
    # Label names come from subfolder names (e.g., ['Alex', 'Kelly'] in sorted order)
    label_names = raw.features["label"].names
    print(label_names)
    num_labels = len(label_names)
    assert num_labels == 2, (
        f"Expected 2 classes, found {num_labels}: {label_names}"
    )

    split = raw.train_test_split(test_size=0.2, stratify_by_column="label", seed=0)
    return label_names, num_labels, split


@app.cell
def _(AutoImageProcessor, T, split):
    train_ds, val_ds = split["train"], split["test"]
    # 2) Image processor + transforms aligned with ViT pretraining
    # checkpoint = "google/vit-base-patch16-224-in21k" # not working--cached issue?
    checkpoint = "microsoft/swin-base-patch4-window7-224"
    # checkpoint = "facebook/deit-base-distilled-patch16-224" # needs TF
    # checkpoint = "microsoft/beit-base-patch16-224-pt22k" # needs Jax
    # checkpoint = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    size = processor.size.get("height", 224)

    normalize = T.Normalize(mean=processor.image_mean, std=processor.image_std)

    train_tfms = T.Compose(
        [
            T.RandomResizedCrop(size, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]
    )

    val_tfms = T.Compose(
        [
            T.Resize(int(size * 1.14)),
            T.CenterCrop(size),
            T.ToTensor(),
            normalize,
        ]
    )


    def _transform_train(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        return {
            "pixel_values": [train_tfms(img) for img in images],
            "labels": batch["label"],
        }


    def _transform_val(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        return {
            "pixel_values": [val_tfms(img) for img in images],
            "labels": batch["label"],
        }


    train_ds = train_ds.with_transform(_transform_train)
    val_ds = val_ds.with_transform(_transform_val)
    return checkpoint, processor, train_ds, val_ds, val_tfms


@app.cell
def _(
    AutoModelForImageClassification,
    checkpoint,
    label_names,
    num_labels,
    os,
):
    # 3) Model head adapted to 2 classes with correct id↔label mapping
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    model = AutoModelForImageClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    clean_checkpoint = checkpoint.replace("/", "_")
    save_point = "weights/" + clean_checkpoint
    os.makedirs(save_point, exist_ok=True) 
    # Freeze the backbone (Swin’s module is `model.swin`, not `model.vit`)
    # backbone = None
    # for attr in ("swin", "vit", "convnext"):
    #     if hasattr(model, attr):
    #         backbone = getattr(model, attr)
    #         break
    # if backbone is None:
    #     raise RuntimeError("Could not locate backbone submodule to freeze.")
    # for p in backbone.parameters():
    #     p.requires_grad = False
    return clean_checkpoint, model, save_point


@app.cell
def _(
    DefaultDataCollator,
    EarlyStoppingCallback,
    PIN_MEM,
    Trainer,
    TrainingArguments,
    evaluate,
    model,
    np,
    processor,
    save_point,
    train_ds,
    val_ds,
):
    # 4) Metrics
    acc = evaluate.load("accuracy")
    f1 = evaluate.load("f1")


    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        return {
            "accuracy": acc.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1_macro": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
        }


    # 5) Training
    args = TrainingArguments(
        output_dir=save_point,
        remove_unused_columns=False,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=100,  # runaway max epochs--real stopping condition is an increase in validation loss
        weight_decay=0.01,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",  # monitor validation accuracy
        greater_is_better=True,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=PIN_MEM,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=4,  # stop if no improvement for 4 evals
                early_stopping_threshold=0.0,  # require strictly better (set >0 to require margin)
            )
        ],
    )
    return (trainer,)


@app.cell
def _(trainer):
    trainer.train()
    return


@app.cell
def _(clean_checkpoint, json, os, processor, save_point, trainer, val_ds):
    mets = trainer.evaluate(eval_dataset=val_ds)

    os.makedirs(f"predictions/{clean_checkpoint}/", exist_ok=True)
    with open(f"predictions/{clean_checkpoint}/metrics.json", "w") as fi:
        json.dump(mets, fi, indent=2)

    trainer.args.save_safetensors = False
    # 6) Save for inferencee
    trainer.save_model(
        save_point
    )  # saves model + config (incl. id2label/label2id)
    processor.save_pretrained(save_point)
    return


@app.cell
def _(
    AutoImageProcessor,
    AutoModelForImageClassification,
    clean_checkpoint,
    csv,
    load_dataset,
    os,
    save_point,
    trainer,
    val_tfms,
):
    model_eval = AutoModelForImageClassification.from_pretrained(save_point).eval()
    processor_eval = AutoImageProcessor.from_pretrained(
        save_point
    )

    hold = load_dataset("imagefolder", data_dir="Data/HoldoutSet01", split="train")

    filenames = [os.path.basename(im.filename) for im in hold["image"]]

    hold = hold.with_transform(
        lambda b: {
            "pixel_values": [val_tfms(im.convert("RGB")) for im in b["image"]]
        }
    )

    logits = trainer.predict(hold).predictions
    pred_ids = logits.argmax(-1)

    id2label_holdout = getattr(model_eval.config, "id2label", {})

    out_path = f"predictions/{clean_checkpoint}/holdout_predictions.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "prediction"])
        for k, y in zip(filenames, pred_ids.tolist()):
            w.writerow([k, id2label_holdout.get(int(y), str(int(y)))])

    print(f"Wrote {len(filenames)} rows to {out_path}")
    return


if __name__ == "__main__":
    app.run()
