# bot/Dockerfile

# Use a base Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy the bot code and requirements file
COPY bot_DIGIBUILD.py /app/bot_DIGIBUILD.py
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Run the bot script
CMD ["python", "bot_DIGIBUILD.py"]
