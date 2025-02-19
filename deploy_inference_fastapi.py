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
                command="gunicorn -w 4 -k uvicorn.workers.UvicornWorker uvicorn app_fastapi_load_test:app --bind 0.0.0.0:8001",
                dockerfile_path= "Diabetic-Retinopathy/src/app/Dockerfile_clientapp",
                build_context_path= "Diabetic-Retinopathy/src/app"
            ),
            build_source=LocalSource(local_build=False),
        ),
        resources=Resources(cpu_request=1, cpu_limit=4, memory_request=2000, memory_limit=4000),
        env={
            "TRITONSERVER_URL": "ophtha-deployment.apps.wadhwaniai.org/optha-inference-service-8000/",
            "SSL": True,
            "UVICORN_WEB_CONCURRENCY": "1",
            "ENVIRONMENT": "dev"
        },
        ports=[
            Port(
                port=8001,
                host="ophtha-deployment.apps.wadhwaniai.org",
                path="/optha-fastapi-service-8001/",
            )
        ],
    )

    service.deploy("production-apso-cp:ophtha-deployment", wait=False)


if __name__ == '__main__':
    deploy_fastapi_service()
