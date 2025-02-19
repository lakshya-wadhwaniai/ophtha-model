from servicefoundry import (
    Service,
    DockerFileBuild,
    Build,
    Resources,
    Port,
    LocalSource,
)


def deploy_inference_service(model_fqn, gradability_model_fqn):
    service = Service(
        name="optha-inference",
        image=Build(
            build_spec=DockerFileBuild(
                command="tritonserver --model-repository model_repository",
                dockerfile_path= "Diabetic-Retinopathy/Dockerfile",
                build_context_path= "Diabetic-Retinopathy"
            ),
            build_source=LocalSource(local_build=False),
        ),
        resources=Resources(cpu_request=0.01, cpu_limit=4, memory_request=3000, memory_limit=6000, shared_memory_size=1000),
        env={
            "INFERENCE_MODEL_FQN": model_fqn,
            "GRADABILITY_MODEL_FQN": gradability_model_fqn,
        },
        ports=[
            Port(
                port=8000,
                host="ophtha-prod-deployment.apps.ehealth.wadhwaniai.org",
                path="/optha-inference-service-8000/",
            )
        ],
    )

    service.deploy("tfy-apso1-tf-oph-ehealth:optha-prod-deployment", wait=False)

if __name__ == '__main__':
    deploy_inference_service("model:wadhwaniai/ophtha-deployment/multiclass-efficientnetv2:5", "model:wadhwaniai/ophtha-deployment/gradability-efficientnetv2:4")
