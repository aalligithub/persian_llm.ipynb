from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "tiiuae/falcon-rw-1b"  # مدل نیم ما بسیار مهم است زیرا مدلی که اینجا وارد کنیم همان مدلی است که آموزش داده میشود به عنوان پیش فرض کوچک ترین مدل برای سرعت بیشتر ران کردن اضافه شده

# گزینه های بیشتر برای مدل:
# tiiuae/falcon-rw-7b
# tiiuae/falcon-mamba-7b-instruct
# The Falcon-180B pretrained and chat models, under the Falcon-180B TII license. Falcon-180B is the largest and most powerful open-access model available.
# The Falcon-7/40B pretrained and instruct models, under the Apache 2.0 software license . Falcon-7B/40B models are state-of-the-art for their size, outperforming other open-source models on NLP benchmarks.

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# دیتا هایی که دانلود کردیم را اینجا اضافه میکنیم اگر با نوت بوک پیشروی میکنید پس از دریافت فایل ها از گیت هاب و اپلود در اینجا مسیر گذاری میکنیم بصورت دیفالت اگر فایل های دانلودی را در روت بگذارید مسیر درست است
train_dataset = load_dataset('conll2003', data_files={'train': '/content/fa_seraji-ud-train.conllu'}, split='train')
dev_dataset = load_dataset('conll2003', data_files={'validation': '/content/fa_seraji-ud-dev.conllu'}, split='validation')
test_dataset = load_dataset('conll2003', data_files={'test': '/content/fa_seraji-ud-test.conllu'}, split='test')


# توکن سازی
def tokenize_function(examples):
    tokens = tokenizer(examples['tokens'], is_split_into_words=True, truncation=True, padding='max_length', max_length=256)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_dev = dev_dataset.map(tokenize_function, batched=True)

# تنظیمات برای آموزش بصورت دیفالت سبک ترین حالت ممکن اما کمی کند تر
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,  # Smaller batch size for lower memory usage
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Simulate a larger batch size
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    optim="adamw_torch_fused",
    dataloader_pin_memory=False,
    fp16=True  # Mixed precision to reduce memory usage
)

model.gradient_checkpointing_enable()
# فعال سازی آموزش دهنده
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev
)



# آموزش نهایی
trainer.train()

# ارزش گذاری مدل
results = trainer.evaluate()
print(results)

# ذخیره سازی مدل و توکن ساز
model.save_pretrained("./falcon-farsi")
tokenizer.save_pretrained("./falcon-farsi")