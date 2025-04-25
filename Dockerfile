# Use an official Python image
FROM python:3.11-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    python3-dev \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python3 - <<EOF
import nltk
# download both the standard punkt and the punkt_tab variant
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
EOF


# Copy application code
COPY . .

# Install Node dependencies and build
RUN npm install && npm run build

# Expose the port Render will bind to
EXPOSE 10000

# Start the app using the Render-assigned port
CMD npm run start
