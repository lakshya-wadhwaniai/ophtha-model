from servicefoundry import (
    Service,
    DockerFileBuild,
    Build,
    Resources,
    Port,
    LocalSource,
)
import argparse


def deploy_fastapi_service():
    service = Service(
        name="optha-fastapi",
        image=Build(
            build_spec=DockerFileBuild(
                command="gunicorn -w 6 -k uvicorn.workers.UvicornWorker app_fastapi_grad_test_cropped:app --bind 0.0.0.0:8001",
                dockerfile_path= "Diabetic-Retinopathy/src/app/Dockerfile_clientapp",
                build_context_path= "Diabetic-Retinopathy/src/app"
            ),
            build_source=LocalSource(local_build=False),
        ),
        resources=Resources(cpu_request=0.01, cpu_limit=4, memory_request=1000, memory_limit=2000),
        env={
            
            "TRITONSERVER_URL": "http://localhost:8000/",
            "SSL": False,
            "UVICORN_WEB_CONCURRENCY": "1",
            "ENVIRONMENT": "dev"
        },
        ports=[
            Port(
                port=8001,
                host="ophtha-prod-deployment.apps.ehealth.wadhwaniai.org",
                path="/optha-fastapi-service-8001/",
            )
        ],
    )

    service.deploy("tfy-apso1-tf-oph-ehealth:optha-prod-deployment", wait=False)


if __name__ == '__main__':
    deploy_fastapi_service()
