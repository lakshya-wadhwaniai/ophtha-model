from servicefoundry import Job, PythonBuild, Schedule, Build, Resources, LocalSource

job = Job(
    # CHANGE NAME
    name="trainjob",
    image=Build(
        # CHANGE PYTHON VERSION AND COMMAND
        build_spec=PythonBuild(
            python_version="3.9.12",
            command="PYTHONPATH='./Diabetic-Retinopathy/' python src/main/train.py --config eyepacs/multiclass_model.yml",
            requirements_path="requirements.txt",
            build_context_path="Diabetic-Retinopathy",
        ),
        build_source=LocalSource(local_build=False),
    ),
    resources=Resources(
        cpu_request=4, cpu_limit=8, memory_request=10000, memory_limit=20000,  ephemeral_storage_limit= 20, ephemeral_storage_request=10,  shared_memory_size=500
    ),
    # env={"TFY_HOST": "https://wadhwaniai.truefoundry.com/", "TFY_API_KEY": "tfy-secret://wadhwaniai:optha-poc:tfyapikey"},
    # env={
    #     "AWS_DEFAULT_REGION": "tfy-secret://wadhwaniai:optha-poc:AWS_DEFAULT_REGION",
    #     "AWS_ACCESS_KEY_ID": "tfy-secret://wadhwaniai:optha-poc:AWS_ACCESS_KEY_ID",
    #     "AWS_SECRET_ACCESS_KEY": "tfy-secret://wadhwaniai:optha-poc:AWS_SECRET_ACCESS_KEY",
    # },
    # trigger=Schedule(schedule="*/5 * * * *"),
)

# CHANGE WORKSPACE NAME
job.deploy("production-apso-cp:ophtha-training", wait=True)