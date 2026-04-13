# Use official FaceFusion 3.6.0 with CUDA (includes HyperSwap models)
FROM facefusion/facefusion:3.6.0-cuda

# Install runpod SDK
RUN pip3 install --no-cache-dir runpod

# Copy handler
COPY handler.py /opt/facefusion/handler.py

# RunPod serverless entry point
CMD ["python3", "/opt/facefusion/handler.py"]
