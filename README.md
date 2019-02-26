# vehicle-price-prediction
doan gia xe

# run

- ```docker build -t price-prediction```
- on local: ```docker run --rm  --name vehicle-price-prediction -v /Users/vietky/Downloads/Lab/sentiment_analysis/:/app/ -p 9000:9000 -d price-prediction:latest```
- on staging: ```docker run --rm  --name vehicle-price-prediction -p 7777:7777 -d docker.chotot.org/price-prediction:latest```