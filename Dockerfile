# ✅ Use Python 3.10 as base image
FROM python:3.10

# ✅ Set working directory
WORKDIR /app

# ✅ Copy all project files into the container
COPY . /app

# ✅ Install required Python dependencies (No Flask Needed)
RUN pip install --no-cache-dir tensorflow anvil-uplink numpy

# ✅ Start the Anvil Uplink server
CMD ["python", "server.py"]
