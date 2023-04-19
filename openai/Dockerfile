FROM python:3.10

RUN pip install --no-cache \
    gradio \
    llama-index \
    openai \
    PyPDF2 \
    python-dotenv

CMD ["python"]

# docker build -t openai .
# docker run --rm -it -v $PWD:/app -p 8080:8080 openai python /app/app.py

# 1.26GB
