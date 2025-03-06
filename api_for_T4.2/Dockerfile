
# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt
COPY . .
# Expose the port FastAPI will run on
EXPOSE 5000
# Command to run the FastAPI server
CMD ["python", "main_tot.py"]
#CMD ["uvicorn", "main_tot.py:app", "--host", "0.0.0.0", "--port", "8000"]
