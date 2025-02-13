install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
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

hf-login:
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload sarahwei/Heart-Disease-Classification ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload sarahwei/Heart-Disease-Classification ./model /model --repo-type=space --commit-message="Sync model"
	huggingface-cli upload sarahwei/Heart-Disease-Classification ./result /metrics --repo-type=space --commit-message="Sync model"

deploy: hf-login push-hub