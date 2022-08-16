

```
conda create -n ons python=3.8
conda activate ons

pip install lightning
pip install -r requirements.txt

jupyter nbconvert --to script glaucoma-classification.ipynb
```

```
mv .env_demo .env
lightning run app nb_app.py --cloud --open-ui False --name ons
```

```
# Research PDF
python -m lightning run app 0_paper_app.py --cloud --open-ui False --name paper

# Jupyter APP
python -m lightning run app 1_jupyter_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env DEMO_AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --env DEMO_AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
--cloud --open-ui false --name jupyter

# Grado APP / Done
python -m lightning run app 2_grado_app.py --env WANDB_API_KEY=$WANDB_API_KEY  --cloud --open-ui false --name grado

# Trainer APP
python -m lightning run app 3_trainer_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env DEMO_AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --env DEMO_AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
--cloud --open-ui false --name trainer

# Demo
python -m lightning run app 4_demo_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env DEMO_AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --env DEMO_AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
--cloud --open-ui false --name demo
```