FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Системные зависимости
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    build-essential \
    git \
    supervisor \
    curl \
    && apt-get clean

# Установка pip и poetry
RUN python3.10 -m pip install --upgrade pip
RUN pip install poetry poetry-plugin-export pyasynchat

# Копируем код
COPY . /code
WORKDIR /code

# Настройка poetry под python3.10
RUN poetry env use python3.10
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
RUN poetry run pip install --no-cache-dir --upgrade -r requirements.txt

# Копируем конфигурацию supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Открываем порты
EXPOSE 3333 8000 8001

# Запуск supervisor
ENTRYPOINT ["/usr/bin/supervisord"]
