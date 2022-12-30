FROM python:3.10-slim

COPY ./requirements.txt /root/server/requirements.txt

RUN chown -R root:root /root/server

WORKDIR /root/server
RUN pip3 install -r requirements.txt

COPY ./ ./
RUN chown -R root:root ./

ENV SECRET_KEY snowden

RUN chmod +x run.py
CMD ["python3", "run.py"]

