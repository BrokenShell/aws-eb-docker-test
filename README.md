## Local Build (not required for AWS)
```
docker build . -t aws-test
```


## Local Run (not required for AWS)
```
docker run -it -p 5000:5000 aws-test uvicorn app.main:app --host=0.0.0.0 --port=5000
```


## EB Configuration
```
eb init
```
Select a default region
- `1) us-east-1 : US East (N. Virginia)`

Select an application to use
- `22) [ Create new Application ]`

Enter Application Name
- `aws-test` - name it whatever you like

Select a platform branch.
- `1) Docker running on 64bit Amazon Linux 2`


## EB Deploy AWS (initial deployment)
```
eb create
```


## EB Deploy AWS (updates)
```
eb deploy
```


## EB Run Remote Web App
```
eb run
```
