FROM python:3.8

# Make a directory
ADD main.py .

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt


# Run
CMD ['python', 'main.py']