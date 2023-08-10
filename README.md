# Model deployment project
[DRAFT]

## Steps I took
Steps

* I created a firewall rule to access an MLflow tracking server
* I created a compute engine instance with 2 vCPUs and 4GB memory
* I created a postgreSQL instance to store the MLflow metadata
* I created a GCS Bucket to store the artifacts
* On the compute instance I installed


mlflow server \
 -h 0.0.0.0 \
 -p 5000 \
 --backend-store-uri postgresql://mlflow-user:'<password>'@<uri_sql_server>/mlflow-db \
 --default-artifact-root gs://mlflow-artifacts/default


 Accessible via: http://35.246.210.0:5000 


 * I started the MLflow server with the compute instance URI and set-up an experiment

![](./images/hyperparameter_comparison.png)


![](./images/parameter_importance.png)


Built the fastapi app in [webservice](./webservice/)

Open up an artifact registry 

Built and pushed the docker container to the registry 
```bash
docker buildx build --no-cache --platform linux/amd64 --push -t europe-west3-docker.pkg.dev/ml-neuefische/docker-registry/webservice-taxi-prediction:latest  .
``````

# Model deployment project

## Dataset

Yellow taxi dataset: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

- Year 2021
- Month 01


## Answer these questions

1. What is the RMSE of your model?
2. What would you do differently if you had more time?


## How to submit your project

Upload your project on GitHub and send us the link. Answer the questions above in the README.md file.
