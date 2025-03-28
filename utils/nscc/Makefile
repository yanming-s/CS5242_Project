d ?= ./
o ?= out/
b ?= $(o)build/

DEVCONTAINER_TAG := 25.01-py3
PROJECT_ID ?= 11004056

pbs_container_name := nvidia-pytorch

devcontainer_sqsh := $(o)nvidia-pytorch-$(DEVCONTAINER_TAG).sqsh

PBS_COMMON_PARAMS += -P $(PROJECT_ID)
PBS_COMMON_PARAMS += -l container_image=$(abspath $(devcontainer_sqsh))
PBS_COMMON_PARAMS += -l container_name=$(pbs_container_name)
PBS_COMMON_PARAMS += -v CONTAINER_NAME=$(pbs_container_name)

$(devcontainer_sqsh): 
	@mkdir -p $(@D)
	enroot import -o $@ docker://nvcr.io/nvidia/pytorch:$(DEVCONTAINER_TAG)

.PHONY: devcontainer
devcontainer: $(devcontainer_sqsh)

.PHONY: submit-%
submit-%: $(d)scripts/%.pbs $(devcontainer_sqsh)
	qsub $(PBS_COMMON_PARAMS) $< 
