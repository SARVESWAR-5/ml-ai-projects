from transformers import pipeline

# Load summarizer pipeline
summarizer = pipeline("summarization")

text = """Machine learning is a method of data analysis that automates analytical model building. 
It is a branch of artificial intelligence based on the idea that systems can learn from data, 
identify patterns and make decisions with minimal human intervention."""

summary = summarizer(text, max_length=40, min_length=10, do_sample=False)
print("Summary:", summary[0]['summary_text'])
