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
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.emal $(USER_EMAIL)
	git commit -am "Update with new result"
	git push --force origin HEAD:update