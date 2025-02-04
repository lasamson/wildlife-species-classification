FROM python:3.12.8-slim

WORKDIR /app

RUN python3 -m pip install flask
RUN python3 -m pip install requests
RUN python3 -m pip install --upgrade Pillow
RUN python3 -m pip install tensorflow==2.18.0
RUN python3 -m pip install tensorflow-serving-api==2.18.0
RUN python3 -m pip install gunicorn

COPY ["scripts/gateway.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "gateway:app"]