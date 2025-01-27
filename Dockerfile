FROM public.ecr.aws/lambda/python:3.11

RUN python3 -m pip install --upgrade Pillow
RUN python3 -m pip install requests
RUN python3 -m pip install tensorflow

COPY bin/custom/custom_dropout_0.5_100_0.846_0.521.keras .
COPY scripts/lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]