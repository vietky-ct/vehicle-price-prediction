FROM python:3.6.8-stretch
RUN pip install --upgrade pip
RUN mkdir /app
WORKDIR /app
# RUN ls
COPY ./requirements.txt /app
RUN pip install -r ./requirements.txt
COPY ./ /app

# ENTRYPOINT ["bash"]