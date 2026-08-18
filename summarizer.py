from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/bart-large-cnn"


print("Loading summarization model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Summarization model loaded!")


def summarize_text(text):

    if not text or not text.strip():
        return "No text was detected."

    # Remove excessive spaces
    text = " ".join(text.split())

    # Keep reasonable amount of text
    words = text.split()

    text = " ".join(words[:1000])

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        min_length=30,
        num_beams=4,
        length_penalty=1.0,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    # Split into sentences
    sentences = []

    for sentence in summary.replace(
        "!", "."
    ).replace(
        "?", "."
    ).split("."):

        sentence = sentence.strip()

        if sentence:
            sentences.append(sentence)

    # Return maximum 4 lines
    return "\n".join(
        sentences[:4]
    )