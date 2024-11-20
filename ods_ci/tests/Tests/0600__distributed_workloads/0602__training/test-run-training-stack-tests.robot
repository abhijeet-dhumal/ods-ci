*** Settings ***
Documentation     Training operator E2E tests - https://github.com/opendatahub-io/distributed-workloads/tree/main/tests/kfto/core
Suite Setup       Prepare Training Operator E2E Core Test Suite
Suite Teardown    Teardown Training Operator E2E Core Test Suite
Library           OperatingSystem
Library           Process
Resource          ../../../../tasks/Resources/RHODS_OLM/install/oc_install.robot
Resource          ../../../../tests/Resources/Page/DistributedWorkloads/DistributedWorkloads.resource


*** Variables ***
${CUDA_TRAINING_IMAGE}            quay.io/modh/training@sha256:b98e373a972ff6f896a9dc054d56920e915675339c02ea7fa123e0f4bbef4d74 
${ROCM_TRAINING_IMAGE}            quay.io/modh/training@sha256:2efb6efba4ec08e63847d701e3062a5f6ddf51c91af5fbcef6378b9e6520a3bb
  

*** Test Cases ***
Run Training operator KFTO test with NVIDIA CUDA image
    [Documentation]    Run Go KFTO tests for Training operator using PyTorch job with NVIDIA CUDA image
    [Tags]  Resources-GPU    NVIDIA-GPUs
    ...     RHOAIENG-16035
    ...     Tier1
    ...     DistributedWorkloads
    ...     Training
    ...     TrainingOperator
    Run Training Operator KFTO Test    TestPyTorchJobWithCuda    ${CUDA_TRAINING_IMAGE}

Run Training operator KFTO test with AMD ROCm image
    [Documentation]    Run Go KFTO tests for Training operator using PyTorch job with AMD ROCm image
    [Tags]  Resources-GPU    AMD-GPUs    ROCm
    ...     RHOAIENG-16035
    ...     Tier1
    ...     DistributedWorkloads
    ...     Training
    ...     TrainingOperator
    Run Training Operator KFTO Test    TestPyTorchJobWithROCm    ${ROCM_TRAINING_IMAGE}

Run Training operator KFTO error handling test with NVIDIA CUDA image
    [Documentation]    Run Go KFTO error handling tests for Training operator using PyTorch job with NVIDIA CUDA image
    [Tags]  RHOAIENG-14542
    ...     Tier1
    ...     DistributedWorkloads
    ...     Training
    ...     TrainingOperator
    Run Training Operator KFTO Test    TestPyTorchJobFailureWithCuda    ${CUDA_TRAINING_IMAGE}

Run Training operator KFTO error handling test with AMD ROCm image
    [Documentation]    Run Go KFTO error handling tests for Training operator using PyTorch job with AMD ROCm image
    [Tags]  RHOAIENG-14542
    ...     Tier1
    ...     DistributedWorkloads
    ...     Training
    ...     TrainingOperator
    Run Training Operator KFTO Test    TestPyTorchJobFailureWithROCm    ${ROCM_TRAINING_IMAGE}
