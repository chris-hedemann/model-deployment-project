# Model deployment project
[DRAFT]

## Steps I took
Steps

* I created a firewall rule to access an MLflow tracking server
* I created a compute engine instance with 2 vCPUs and 4GB memory
* I created a postgreSQL instance to store the MLflow metadata
* I created a GCS Bucket to store the artifacts
* On the compute instance I installed

sudo apt-get update
sudo apt-get install git python3-pip make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
curl https://pyenv.run | bash
pyenv install 3.10.11
pyenv global 3.10.11
python -m venv mlflow
source mlflow/bin/activate
pip install mlflow boto3 google-cloud-storage psycopg2-binary

mlflow server \
 -h 0.0.0.0 \
 -p 5000 \
 --backend-store-uri postgresql://mlflow-user:'<password>'@<uri_sql_server>/mlflow-db \
 --default-artifact-root gs://mlflow-artifacts/default


 Accessible via: http://35.246.210.0:5000 


 * I started the MLflow server with the compute instance URI and set-up an experiment

![](./images/hyperparameter_comparison.png)



![](./images/parameter_importance.png)
