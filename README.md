### Introduction
Lightning App is composed of Lightning Work and Lightning Flow. Start by wrapping existing scripts as Lightning Works. Lightning Works send state information to Lighting Flows. Lightning Flows send run command to Lightning Works. Distributed states and runs are serialized via event loops in Lightning Flows.

```mermaid
graph BT;
  subgraph Local VM
    LF((App <br><br>Lightning <br>Flow))
    T(Train <br><br>Lighting Work)      -- state <br>changes --> LF
    I(Inference <br><br>Lightning Work) -- state <br>changes --> LF
    D(Diag <br><br>Lightning Work)      -- state <br>changes --> LF
    U(UI <br><br>Lightning FLow)        -- state <br>changes --> LF  
    LF -- run --> T
    LF -- run --> I
    LF -- run --> D 
    LF -- run --> U 
    subgraph existing scripts
      TS[train_script.py]
      IS[gradio_script.py]
      DS[tensorboard]
      US[ui_script.py]
    end
    subgraph wrapper code
      T ---> TS
      I ---> IS
      D ---> DS
      U ---> US  
    end
  end
```

This is a demo lightning app that gradually shows us how to build applications step by step. Make sure that you execute the commands below. 

```
conda create -n ons python=3.8
conda activate ons

pip install lightning
pip install -r requirements.txt

# Optional - Convert Jupyter Notebook to python script
jupyter nbconvert --to script glaucoma-classification.ipynb
```

```
mv .env_demo .env
lightning run app nb_app.py --cloud --open-ui False --name ons
```

Please execute the appications below serially as the complexity level varies. The final app `demo` shows how applications can organised in a sequence and executed serially. We also see lighthing play well with 3rd party components like `W&B`, `Grado` etc.

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

# Demo (Training and Inference)
python -m lightning run app 4_demo_app.py --env WANDB_API_KEY=$WANDB_API_KEY \
--env DEMO_AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID --env DEMO_AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
--cloud --open-ui false --name demo
```