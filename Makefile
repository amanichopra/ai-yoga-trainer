venv:
	python3 -m venv .venv --prompt ai-yoga-trainer

generate_requirements:
	pip3 freeze > requirements.txt

gcloud-setup:
	gcloud config set project applied-ml-413816
	gcloud auth login
	gcloud auth application-default login	

setup:
	.venv/bin/pip install -U pip setuptools
	.venv/bin/pip install -U -r requirements.txt