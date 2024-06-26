install:
	pip install --upgrade pip &&\
		pip install -r requriements.txt
format:
	black *.py
train:
	python train.py
eval:
	echo "## Model Metrics" > report.md
	cat ./result/metrics.txt >> report.md

	echo '\n ## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./result/model_results.png)' >> report.md

	cml comment create report.md