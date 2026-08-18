from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "facebook/bart-large-cnn"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")


def summarize_text(text):

    if not text or not text.strip():
        return "No text was detected."

    # Limit input size
    words = text.split()
    text = " ".join(words[:1000])

    # Convert text into tokens
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=100,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    # Split into sentences
    sentences = (
        summary
        .replace("!", ".")
        .replace("?", ".")
        .split(".")
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    # Maximum 4 lines
    return "\n".join(sentences[:4])


# Test
if __name__ == "__main__":

    test_text = """
    Artificial intelligence is transforming many industries around
    the world. Organizations use artificial intelligence to automate
    repetitive tasks, analyze large amounts of information and make
    better decisions. Machine learning allows computers to identify
    patterns in data and make predictions without being explicitly
    programmed. AI is now being used in healthcare, finance,
    education, transportation and many other fields.
    """

    print("\n========== SUMMARY ==========\n")

    print(summarize_text(test_text))